#include "fused_odometry/radar_processor.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>
#include <algorithm>

namespace fused_odometry {

RadarProcessor::RadarProcessor(const Config& cfg)
    : config_(cfg), rng_(std::random_device{}())
{
    updateRansacIterations();
}

bool RadarProcessor::estimate(const sensor_msgs::msg::PointCloud2& cloud_msg,
                              double pitch, double roll, double /*yaw*/, bool holonomic,
                              Eigen::Vector3d& v_r, Eigen::Vector3d& sigma,
                              sensor_msgs::msg::PointCloud2& inliers_msg,
                              sensor_msgs::msg::PointCloud2& outliers_msg,
                              sensor_msgs::msg::PointCloud2& dynamic_msg,
                              const Eigen::Quaterniond& q_world)
{
    // Convert from ROS to PCL
    pcl::PointCloud<RadarPointCloudType> cloud;
    pcl::fromROSMsg(cloud_msg, cloud);

    // Accept all finite points — the upstream preprocessing node has already
    // applied range, RCS, and FoV filtering.  Re-filtering here would reject
    // points whose coordinates were flipped by the preprocessing node
    // (e.g. flip_x inverts azimuth, causing forward-facing points to appear
    // backward and fail a second FoV check).
    std::vector<RadarPointCloudType> pts;
    pts.reserve(cloud.size());

    for (const auto& p : cloud) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
            !std::isfinite(p.z) || !std::isfinite(p.v)) continue;
        double r = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        if (r < 1e-6) continue;  // degenerate point
        pts.push_back(p);
    }

    // Derotate points to world frame if configured.
    // Rotating positions by IMU orientation before RANSAC makes the Doppler
    // geometry solve for v_world directly (Doppler scalar is invariant to
    // rotation since both numerator and denominator rotate together).
    if (config_.points_in_world_frame) {
        const Eigen::Matrix3d R = q_world.toRotationMatrix();
        for (auto& p : pts) {
            Eigen::Vector3d pw = R * Eigen::Vector3d(p.x, p.y, p.z);
            p.x = static_cast<float>(pw.x());
            p.y = static_cast<float>(pw.y());
            p.z = static_cast<float>(pw.z());
            // Doppler velocity (p.v) is a scalar — no rotation needed
        }
        // When points are world-frame, force holonomic solve (non-holonomic
        // forward-only constraint is a body-frame concept)
        holonomic = true;
    }

    // ── Dynamic object removal ─────────────────────────────────────────────
    // Remove points whose Doppler is inconsistent with last frame's ego-velocity.
    // For a static reflector: measured_doppler ≈ -h_i · v_prev (within thresh).
    // Dynamic objects violate this; we discard them before RANSAC runs.
    pcl::PointCloud<RadarPointCloudType> dynamic_cloud;
    if (config_.dynamic_removal_thresh > 0.0 && v_prev_valid_) {
        std::vector<RadarPointCloudType> static_pts;
        static_pts.reserve(pts.size());
        for (const auto& p : pts) {
            double r = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
            // Unit LOS vector
            Eigen::Vector3d h(p.x / r, p.y / r, p.z / r);
            // Predicted Doppler for a static object given previous ego-velocity
            double d_pred = -h.dot(v_prev_) * config_.doppler_velocity_correction_factor;
            double d_meas = -static_cast<double>(p.v) * config_.doppler_velocity_correction_factor;
            double residual = std::abs(d_meas - d_pred);
            if (residual > config_.dynamic_removal_thresh) {
                dynamic_cloud.push_back(p);   // classified as dynamic → removed
            } else {
                static_pts.push_back(p);
            }
        }
        // Safety: if we filtered too aggressively and too few points remain,
        // skip filtering this frame and use all points (avoids false positives
        // during aggressive maneuvers or on the very first good keyframe).
        if (static_pts.size() >= 3) {
            pts = std::move(static_pts);
            RCLCPP_DEBUG_STREAM(rclcpp::get_logger("radar_processor"),
                "Dynamic filter: removed " << dynamic_cloud.size()
                << " / " << (pts.size() + dynamic_cloud.size()) << " points");
        } else {
            dynamic_cloud.clear();  // don't publish spurious dynamic cloud
        }
    }
    // Build dynamic_msg regardless (empty if filter disabled or fell back)
    pcl::toROSMsg(dynamic_cloud, dynamic_msg);
    dynamic_msg.header = cloud_msg.header;

    if (pts.empty()) return false;

    // Check for near-zero motion
    double mean_doppler = 0.0;
    for (const auto& p : pts) mean_doppler += std::abs(p.v);
    mean_doppler /= static_cast<double>(pts.size());

    if (mean_doppler < 0.10) {
        v_r   = Eigen::Vector3d::Zero();
        sigma = Eigen::Vector3d(0.03, 0.03, 0.05);
        buildInlierOutlierMsgs(pts, {}, cloud_msg.header, inliers_msg, outliers_msg);
        dynamic_msg.header = cloud_msg.header;   // ADD: empty dynamic cloud
        return true;
    }

    // Build observation matrix H and Doppler vector d
    const size_t N = pts.size();
    Eigen::MatrixXd H(N, 3);
    Eigen::VectorXd d(N);
    for (size_t i = 0; i < N; ++i) {
        double r = std::sqrt(pts[i].x*pts[i].x + pts[i].y*pts[i].y + pts[i].z*pts[i].z);
        H(i, 0) = pts[i].x / r;
        H(i, 1) = pts[i].y / r;
        H(i, 2) = pts[i].z / r;
        d(i)    = -pts[i].v * config_.doppler_velocity_correction_factor;
    }

    // RANSAC
    RadarEgoVelState state;
    ransac(pts, H, d, pitch, roll, holonomic, state);

    const double allowed_outliers = 0.50 * static_cast<double>(N);
    if (state.best_inliers.size() < 3 ||
        static_cast<double>(N - state.best_inliers.size()) > allowed_outliers) {
        dynamic_msg.header = cloud_msg.header;   // ADD: empty dynamic cloud
        return false;
    }

    // Least-squares refinement on inliers
    const size_t n_in = state.best_inliers.size();
    Eigen::MatrixXd Hin(n_in, 3);
    Eigen::VectorXd din(n_in);
    for (size_t k = 0; k < n_in; ++k) {
        size_t idx = state.best_inliers[k];
        Hin.row(k) = H.row(idx);
        din(k)     = d(idx);
    }

    Eigen::Vector3d v_ls;
    if (holonomic) {
        v_ls = (Hin.transpose() * Hin).ldlt().solve(Hin.transpose() * din);
    } else {
        v_ls = solveNonHolonomic(Hin, din);
    }

    // ── Upgrade 3: Per-point 2σ Doppler gate (tighten inliers) ────────
    // After LS refinement, check each inlier point to see if its Doppler
    // prediction matches the refined velocity estimate within 2σ.
    // This removes outliers that the original RANSAC threshold missed.
    {
        std::vector<size_t> tightened_inliers;
        Eigen::VectorXd res = din - Hin * v_ls;

        for (size_t k = 0; k < n_in; ++k) {
            double residual = std::abs(res(k));
            // Threshold: use RANSAC inlier_thresh as a reference (in m/s)
            // Multiply by 2 to be conservative; adjust if needed
            double threshold = config_.inlier_thresh * 2.0;
            if (residual < threshold) {
                tightened_inliers.push_back(state.best_inliers[k]);
            }
        }

        // Re-solve with tightened set if at least 3 points remain
        if (tightened_inliers.size() >= 3) {
            state.best_inliers = tightened_inliers;
            size_t n_tight = tightened_inliers.size();

            Eigen::MatrixXd Hin_tight(n_tight, 3);
            Eigen::VectorXd din_tight(n_tight);
            for (size_t k = 0; k < n_tight; ++k) {
                size_t idx = tightened_inliers[k];
                Hin_tight.row(k) = H.row(idx);
                din_tight(k) = d(idx);
            }

            // Re-solve LS on tightened set
            if (holonomic) {
                v_ls = (Hin_tight.transpose() * Hin_tight).ldlt().solve(Hin_tight.transpose() * din_tight);
            } else {
                v_ls = solveNonHolonomic(Hin_tight, din_tight);
            }
        }
    }

    v_r = v_ls;

    // Apply axis flips if configured
    if (config_.flip_x) v_r.x() = -v_r.x();
    if (config_.flip_y) v_r.y() = -v_r.y();
    if (config_.flip_z) v_r.z() = -v_r.z();

    // Cache velocity (post-flip) for next frame's dynamic filter.
    // Must be post-flip so the cached frame matches the frame of incoming v_r
    // values on the next call (which are also in the post-flip sensor frame).
    v_prev_      = v_r;
    v_prev_valid_ = true;

    // Adaptive EMA: trust the new measurement less when inlier count is low.
    // Disabled when using a factor graph — the FG already provides optimal
    // smoothing; EMA introduces lag and hides measurement uncertainty.
    if (config_.enable_ema) {
        const double alpha = (state.best_inliers.size() >= config_.inlier_count_thresh)
                             ? config_.alpha_high_inliers : config_.alpha_low_inliers;
        if (!v_smoothed_init_) {
            v_smoothed_ = v_r;
            v_smoothed_init_ = true;
        } else {
            v_smoothed_ = alpha * v_r + (1.0 - alpha) * v_smoothed_;
        }
        v_r = v_smoothed_;
    }

    // Proper covariance from Hessian: (H^T*H)^{-1} * (res²/dof)
    // Use the final inlier set (after tightening) for consistent covariance.
    const size_t n_final = state.best_inliers.size();
    Eigen::MatrixXd H_final(n_final, 3);
    Eigen::VectorXd d_final(n_final);
    for (size_t k = 0; k < n_final; ++k) {
        size_t idx = state.best_inliers[k];
        H_final.row(k) = H.row(idx);
        d_final(k) = d(idx);
    }
    Eigen::VectorXd res_final = d_final - H_final * v_ls;
    Eigen::Matrix3d HtH = H_final.transpose() * H_final;
    if (n_final > 3) {
        Eigen::Matrix3d cov = HtH.ldlt().solve(Eigen::Matrix3d::Identity()) *
                             (res_final.squaredNorm() / static_cast<double>(n_final - 3));
        sigma = cov.diagonal().cwiseSqrt();
    } else {
        sigma = Eigen::Vector3d(0.12, 0.15, 0.20);
    }

    buildInlierOutlierMsgs(pts, state.best_inliers, cloud_msg.header, inliers_msg, outliers_msg);
    return true;
}

Eigen::Vector3d RadarProcessor::solveNonHolonomic(const Eigen::MatrixXd& H,
                                                  const Eigen::VectorXd& d) const
{
    auto svd = H.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.singularValues().minCoeff() < 1e-6)
        return Eigen::Vector3d::Zero();
    return svd.solve(d);
}

void RadarProcessor::ransac(const std::vector<RadarPointCloudType>& /*pts*/,
                            const Eigen::MatrixXd& H,
                            const Eigen::VectorXd& d,
                            double /*pitch*/, double /*roll*/, bool holonomic,
                            RadarEgoVelState& state)
{
    const size_t N = static_cast<size_t>(H.rows());
    const unsigned n_samp = std::min(config_.N_ransac_points, static_cast<unsigned>(N));
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    size_t best_count = 0;
    state.best_v = Eigen::Vector3d::Zero();

    for (unsigned iter = 0; iter < ransac_iterations_; ++iter) {
        // Draw n_samp unique samples
        std::vector<size_t> idx;
        idx.reserve(n_samp);
        while (idx.size() < n_samp) {
            size_t s = dist(rng_);
            if (std::find(idx.begin(), idx.end(), s) == idx.end())
                idx.push_back(s);
        }

        Eigen::MatrixXd Hs(n_samp, 3);
        Eigen::VectorXd ds(n_samp);
        for (size_t k = 0; k < n_samp; ++k) {
            Hs.row(k) = H.row(idx[k]);
            ds(k)     = d(idx[k]);
        }

        Eigen::Vector3d v_cand;
        if (holonomic) {
            if (static_cast<int>(n_samp) < 3) continue;
            auto svd = Hs.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
            if (svd.singularValues().minCoeff() < 1e-6) continue;
            v_cand = svd.solve(ds);
        } else {
            if (static_cast<int>(n_samp) < 3) continue;
            v_cand = solveNonHolonomic(Hs, ds);
        }

        // Count inliers
        Eigen::VectorXd res = (H * v_cand - d).cwiseAbs();
        std::vector<size_t> inliers;
        inliers.reserve(N);
        for (size_t i = 0; i < N; ++i)
            if (res(i) < config_.inlier_thresh) inliers.push_back(i);

        if (inliers.size() > best_count) {
            best_count   = inliers.size();
            state.best_v = v_cand;
            state.best_inliers = inliers;
        }
    }
}

void RadarProcessor::buildInlierOutlierMsgs(
    const std::vector<RadarPointCloudType>& pts,
    const std::vector<size_t>& inlier_idx,
    const std_msgs::msg::Header& header,
    sensor_msgs::msg::PointCloud2& in_msg,
    sensor_msgs::msg::PointCloud2& out_msg)
{
    std::vector<bool> is_inlier(pts.size(), false);
    for (size_t i : inlier_idx) {
        if (i < pts.size()) is_inlier[i] = true;
    }

    pcl::PointCloud<RadarPointCloudType> in_cloud, out_cloud;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (is_inlier[i]) in_cloud.push_back(pts[i]);
        else              out_cloud.push_back(pts[i]);
    }
    pcl::toROSMsg(in_cloud,  in_msg);
    pcl::toROSMsg(out_cloud, out_msg);
    in_msg.header  = header;
    out_msg.header = header;
}

}  // namespace fused_odometry
