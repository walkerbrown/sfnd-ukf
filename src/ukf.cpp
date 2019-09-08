#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;


// Construct the unscented Kalman filter (UKF). Please note the motion model
// assumption: Constant turn-rate and velocity magnitude (CTRV).
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

  is_initialized_ = false;
}


UKF::~UKF() {}


// UKF prediction step, estimate the object's location at next timestep k+1
void UKF::Prediction(double delta_t) {
  
  // STEP 1) Generate sigma (sampling) points
  // cout << "Generating sigma points" << endl;
  
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

  // STEP 2a) Transform the sigma points into the augmented state space
  // cout << "Transforming sigma points into the augmented state space" << endl; 
  
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

  // STEP 2b) Predict the motion of each sigma point
  // cout << "Predicting the motion of each sigma point" << endl; 

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

    // Transform from the augmented to the original (n_x-dim) state space
    
    // Avoid division by zero
    if (psid > 0.0001) {
      Xsig_pred(0, i) = px + v / psid * (sin(psi + psid * dt) - sin(psi)) +
                        pow(dt, 2) / 2 * cos(psi) * nu_a;
      Xsig_pred(1, i) = py + v / psid * (-cos(psi + psid *dt) + cos(psi)) +
                        pow(dt, 2) / 2 * sin(psi) * nu_a;
    } else {
      Xsig_pred(0, i) = px + pow(dt, 2) / 2 * cos(psi) * nu_a;
      Xsig_pred(1, i) = py + pow(dt, 2) / 2 * sin(psi) * nu_a;
    }
    
    Xsig_pred(2, i) = v + 0 + dt * nu_a;
    Xsig_pred(3, i) = psi + psid * dt + pow(dt, 2) / 2 * nu_psidd;
    Xsig_pred(4, i) = psid + 0 + dt * nu_psidd;
  }

  // STEP 3) Predict the next state: mean and covariance matrices
  // cout << "Predicting the next state's mean and covariance" << endl; 
  
  // Set the weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_sig_; i++) {
    weights_(i) = 1 / (2 * (lambda_ + n_aug_));
  }

  // Predict the mean state
  for (int i = 0; i < n_sig_; i++) {
    x_ = x_ + weights_(i) * Xsig_pred.col(i);
  }

  // Predict the covariance matrix
  for (int i = 0; i < n_sig_; i++) {
    P_ = P_ + weights_(i) *
            (Xsig_pred.col(i) - x_) *
            (Xsig_pred.col(i) - x_).transpose();
  }
}


// Branch to a lidar or radar measurement update, depending on sensor type
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    x_.fill(0.0);
    x_(0) = 1.0;
    P_ = MatrixXd::Identity(n_x_, n_x_);
    is_initialized_ = true;
  }

  if (MeasurementPackage::SensorType::LASER == meas_package.sensor_type_) {
    UpdateLidar(meas_package);
  }
  else if (MeasurementPackage::SensorType::RADAR == meas_package.sensor_type_) {
    UpdateRadar(meas_package);
  }
  // Handle invalid sensor types by logging an error and skipping that update
  else {
    cout << "UKF::ProcessMeasurement() ERROR: Invalid sensor type "
              << meas_package.sensor_type_ << endl;
  }
}


// Use lidar data to update the belief about the object's current position
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // STEP 4) Predict the measurement mean and covariance; calculate Kalman gain
  // cout << "LASER Predicting the measurement mean and covariance; calculating Kalman gain" << endl; 
  
  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size(); // Measurement z is a 2x1 vector for lidar
  
  // Measurement matrix
  MatrixXd H = MatrixXd(n_z, n_x_);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;

  // Measurement covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << pow(std_laspx_, 2), 0,
       0, pow(std_laspy_, 2);

  VectorXd z_pred = H * x_;  // Extract the px, py values from the state mean
  VectorXd y = z - z_pred;  // Calculate the residuals vector y
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd Sinv = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Sinv;

  // STEP 5) Update the state, by applying the Kalman gain to the residual
  // cout << "LASER Updating the state by applying the Kalman gain to the residual" << endl; 

  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);

  // Update the mean and covariance matrix
  x_ = x_ + (K * y);
  P_ = (I - K * H) * P_;

  // Calculate normalized innovation squared (NIS) for tuning
  double NIS = y.transpose() * Sinv * y;
  cout << "Lidar NIS (2-df X^2, 95% < 5.991) = " << NIS << endl; 
}


// Use radar data to update the belief about the object's current position
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // STEP 4) Predict the measurement mean (z_pred) and innovation covariance (S)
  // cout << "RADAR Predicting the measurement mean and covariance; calculating Kalman gain" << endl; 
  
  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size(); // Measurement z is a 3x1 vector for radar

  // Setup augmented weights vector
  VectorXd weights_aug = VectorXd(n_sig_aug_);
  double w0 = lambda_ / (lambda_ + n_sig_aug_);
  double w = 1 / (2 * (lambda_ + n_sig_aug_));
  weights_aug.fill(w);
  weights_aug(0) = w0;

  MatrixXd Xsig_pred = MatrixXd(n_x_, n_sig_aug_);
  MatrixXd Zsig = MatrixXd(n_z, n_sig_aug_);
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // Transform the sigma points into measurement space
  for (int i = 0; i < n_sig_aug_; i++) {
      double px = Xsig_pred(0, i);
      double py = Xsig_pred(1, i);
      double v = Xsig_pred(2, i);
      double psi = Xsig_pred(3, i);
      double psid = Xsig_pred(4, i);
      
      double rho = sqrt(pow(px, 2) + pow(py, 2));
      double phi = std::atan2(py, px);
      double rhod = 0.0;
      if (rho > 0.0001) {
        rhod = (px * cos(psi) * v + py * sin(psi) * v) / rho; 
      }

      Zsig(0, i) = rho;
      Zsig(1, i) = phi;
      Zsig(2, i) = rhod;
  }

  // Calculate the predicted mean measurement z_pred
  for (int i = 0; i < n_sig_aug_; i++) {
    z_pred = z_pred + weights_aug(i) * Zsig.col(i);  
  }
  
  // Calculate the innovation covariance matrix S
  MatrixXd R = MatrixXd(n_z, n_z);
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  double std_rho2 = pow(std_radr_, 2);
  double std_phi2 = pow(std_radphi_, 2);
  double std_rhod2 = pow(std_radrd_, 2);

  R << std_rho2, 0, 0,
       0, std_phi2, 0,
       0, 0, std_rhod2;

  for (int i = 0; i < n_sig_aug_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = std::fmod(z_diff(1), 2 * M_PI);  // Angle normalization for phi

    S = S + weights_aug(i) * z_diff * z_diff.transpose();
  }

  S = S + R;

  // STEP 5) Update the state, by applying the Kalman gain to the residual
  // cout << "RADAR Updating the state by applying the Kalman gain to the residual" << endl; 

  // Calculate the cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  for (int i = 0; i < n_sig_aug_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = std::fmod(z_diff(1), 2 * M_PI);  // Angle normalization for phi

    VectorXd x_diff = Xsig_pred.col(i) - x_;
    x_diff(3) = std::fmod(x_diff(3), 2 * M_PI);  // Angle normalization for phi

    Tc += weights_aug(i) * x_diff * z_diff.transpose();
  }

  // Calculate the Kalman gain
  MatrixXd K = MatrixXd(n_x_, n_z); 
  MatrixXd Sinv = S.inverse();
  VectorXd residuals = z - z_pred;
  K = Tc * Sinv;
  
  // Update the mean and covariance matrix
  x_ = x_ + K * residuals;
  P_ = P_ - K * S * K.transpose();

  // Calculate normalized innovation squared (NIS) for tuning
  double NIS = residuals.transpose() * Sinv * residuals;
  cout << "Radar NIS (3-df X^2, 95% < 7.815) = " << NIS << endl; 
}
