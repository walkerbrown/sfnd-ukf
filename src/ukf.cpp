#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 * Please note the motion model assumption:
 *   Constant turn-rate and velocity magnitude (CTRV)
 */

UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  n_x_ = 5;
  n_aug_ = 7;
  n_sig_ = 2 * n_x_ + 1;
  n_sig_aug_ = 2 * n_aug_ + 1;
  lambda_ = 3.0 - n_x_;
  weights_ = VectorXd(n_sig_);

  x_.fill(0.0);
  P_.fill(0.0);
}

UKF::~UKF() {}

// UKF prediction step, estimate the object's location at next timestep k+1
void UKF::Prediction(double delta_t) {
  
  // 1) Generate sigma (sampling) points
  
  MatrixXd Xsig = MatrixXd(n_x_, n_sig_);
  MatrixXd A = P_.llt().matrixL();
  double c = sqrt(lambda_ + n_x_);
  MatrixXd cA = c * A;

  Xsig.col(0) = x_;  // First column of Xsig is central sigma point
  for (int i = 1; i <= n_x_; i++) {
    Xsig.col(i) = x_ + cA.col(i - 1);  // First group of sigma points
  }
  for (int i = n_x_ + 1; i <= n_sig_ - 1; i++) {
    Xsig.col(i) = x_ - cA.col(i - 1 - n_x_);  // Symmetric group of sigma points
  }

  // 2) Transform the sigma points into the augmented state space
  
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_aug_);
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;  // Augment the mean state

  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);  // Augment the covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = pow(std_a_, 2);
  P_aug(n_x_ + 1, n_x_ + 1) = pow(std_yawdd_, 2);

  MatrixXd A_aug = P_aug.llt().matrixL();  // Create the square root matrix A
  double c_aug = sqrt(lambda_ + n_aug_);
  MatrixXd cA_aug = c_aug * A_aug;
  
  Xsig_aug.col(0) = x_aug; // First column of Xsig_aug is central sigma point
  for (int i = 1; i <= n_aug_; i++) {
    Xsig_aug.col(i) = x_aug + cA_aug.col(i - 1);  // First group of sigma points
  }
  for (int i = n_aug_ + 1; i <= n_sig_aug_ - 1; i++) {
    Xsig_aug.col(i) = x_aug - cA_aug.col(i - 1 - n_aug_); // Symmetric group
  }

  // 3) Predict the motion of each sigma point
  
  double dt = delta_t;
  MatrixXd Xsig_pred = MatrixXd(n_x_, n_sig_aug_);
  Xsig_pred.fill(0.0);

  // Loop over each sigma point, transforming back to the original state space
  for (int i = 0; i < n_sig_aug_; i++) {
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i); 
    double psi = Xsig_aug(3, i);
    double psid = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_psidd = Xsig_aug(6, i);
    
    // Avoid division by zero
    if (0 == psid) {
      std::cout << "UKF::ProcessMeasurement() ERROR: divide by zero"
                << std::endl;
      return;  // Skip this prediction step
    }

    // Transform from the augmented to the original (n_x-dim) state space
    Xsig_pred(0, i) = px + v / psid * (sin(psi + psid * dt) - sin(psi)) +
                      pow(dt, 2) / 2 * cos(psi) * nu_a;
    Xsig_pred(1, i) = py + v / psid * (-cos(psi + psid *dt) + cos(psi)) +
                      pow(dt, 2) / 2 * sin(psi) * nu_a;
    Xsig_pred(2, i) = v + 0 + dt * nu_a;
    Xsig_pred(3, i) = psi + psid * dt + pow(dt, 2) / 2 * nu_psidd;
    Xsig_pred(4, i) = psid + 0 + dt * nu_psidd;
  }

  // 4) Predict the next state: mean and covariance matrices
  
  // Set the weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_sig_aug_; i++) {
    weights_(i) = 1 / (2 * (lambda_ + n_aug_));
  }

  // Predict the mean state
  for (int i = 0; i < n_sig_aug_; i++) {
    x_ = x_ + weights_(i) * Xsig_pred.col(i);
  }

  // Predict the covariance matrix
  for (int i = 0; i < n_sig_aug_; i++) {
    P_ = P_ + weights_(i) *
            (Xsig_pred.col(i) - x_) *
            (Xsig_pred.col(i) - x_).transpose();
  }
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // Branch to a lidar or radar measurement update, depending on sensor type
  if (MeasurementPackage::SensorType::LASER == meas_package.sensor_type_) {
    UpdateLidar(meas_package);
  }
  else if (MeasurementPackage::SensorType::RADAR == meas_package.sensor_type_) {
    UpdateRadar(meas_package);
  }
  // Handle invalid sensor types by logging an error and skipping that update
  else {
    std::cout << "UKF::ProcessMeasurement() ERROR: Invalid sensor type "
              << meas_package.sensor_type_ << std::endl;
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}
