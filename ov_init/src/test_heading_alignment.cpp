#include "utils/sensor_data.h"
#include "utils/helper.h"

#include <Eigen/Geometry>

#include <cmath>
#include <iostream>

using ov_init::InitializerHelper;

static bool check_frame(const Eigen::Vector3d &gravity_inI, const Eigen::Vector3d &forward_inI) {
  Eigen::Matrix3d R_GtoI;
  InitializerHelper::gram_schmidt(gravity_inI, R_GtoI, true, forward_inI);
  const Eigen::Vector3d forward_inG = R_GtoI.transpose() * forward_inI;
  return R_GtoI.allFinite() &&
         (R_GtoI.transpose() * R_GtoI).isApprox(Eigen::Matrix3d::Identity(), 1e-12) &&
         R_GtoI.col(2).isApprox(gravity_inI.normalized(), 1e-12) &&
         forward_inG.x() > 0.0 && std::abs(forward_inG.y()) < 1e-12;
}

int main() {
  if (!check_frame(Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d::UnitX()) ||
      !check_frame(Eigen::Vector3d(0.0, 0.0, -1.0), Eigen::Vector3d::UnitX()) ||
      !check_frame(Eigen::Vector3d(0.2, -0.3, 0.93), Eigen::Vector3d::UnitX())) {
    std::cerr << "Startup forward axis did not produce zero yaw in a gravity-aligned frame\n";
    return 1;
  }

  // Dynamic initialization uses the final IMU's forward vector expressed in I0.
  const Eigen::Matrix3d R_I0toIinit = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d forward_inI0 = R_I0toIinit.transpose() * Eigen::Vector3d::UnitX();
  Eigen::Matrix3d R_GtoI0;
  InitializerHelper::gram_schmidt(Eigen::Vector3d::UnitZ(), R_GtoI0, true, forward_inI0);
  const Eigen::Vector3d forward_at_init_inG = (R_I0toIinit * R_GtoI0).transpose() * Eigen::Vector3d::UnitX();
  if (forward_at_init_inG.x() < 0.0 || std::abs(forward_at_init_inG.y()) > 1e-12) {
    std::cerr << "Dynamic initialization did not align the final forward heading\n";
    return 1;
  }

  // A vertical forward axis has no horizontal heading. The fallback must remain a valid rotation.
  Eigen::Matrix3d R_vertical;
  InitializerHelper::gram_schmidt(Eigen::Vector3d(1.0, 0.0, 0.0), R_vertical, true);
  if (!R_vertical.allFinite() ||
      !(R_vertical.transpose() * R_vertical).isApprox(Eigen::Matrix3d::Identity(), 1e-12)) {
    std::cerr << "Vertical forward fallback is not a valid rotation\n";
    return 1;
  }

  return 0;
}
