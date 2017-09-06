/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */

	num_particles = 200;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle p;
		p.id	= i;
		p.x		= dist_x(gen);
		p.y		= dist_y(gen);
		p.theta	= dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(1.0);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */

	for (auto &particle : particles) {
		// Calculate Prediction Step
		if (fabs(yaw_rate) < 0.001) {
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		}
		else {
			particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta));
			particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t));
			particle.theta += yaw_rate * delta_t;
		}
		// add random Gaussian noise
		normal_distribution<double> dist_x(0, std_pos[0]);
		normal_distribution<double> dist_y(0, std_pos[1]);
		normal_distribution<double> dist_theta(0, std_pos[2]);
		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	/**
	 * dataAssociation Finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association).
	 * @param predicted Vector of predicted landmark observations
	 * @param observations Vector of landmark observations
	 */

	double closest_distance, distance;
	for(int i = 0; i < observations.size(); i++) {
		LandmarkObs observation = observations[i];
		for(int j = 0; j < predicted.size(); j++){
			LandmarkObs prediction = predicted[j];
			distance = dist(prediction.x,prediction.y,observation.x,observation.y);
			if(j==0 || distance < closest_distance){
				closest_distance = distance;
				observations[i].id = j;
			}
		}
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the
	 *   observed measurements.
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
	 *   standard deviation of bearing [rad]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 *
	 */

	for(int i=0; i < particles.size(); i++) {
		auto particle = particles[i];

		// filter by sensor_range
		std::vector<LandmarkObs> in_range_landmarks;
		for(auto landmark : map_landmarks.landmark_list) {
			LandmarkObs in_range_landmark;
			if (sensor_range > dist(particle.x, particle.y, landmark.x_f, landmark.y_f))
				in_range_landmark.x = landmark.x_f;
				in_range_landmark.y = landmark.y_f;
				in_range_landmark.id = landmark.id_i;
				in_range_landmarks.push_back(in_range_landmark);
		}

		// Coordinate Transformations
		std::vector<LandmarkObs> observations_map;
		for(auto observation : observations) {
			LandmarkObs observation_map;
			observation_map.x = particle.x + observation.x * cos(particle.theta) - observation.y * sin(particle.theta);
			observation_map.y = particle.y + observation.x * sin(particle.theta) + observation.y * cos(particle.theta);
			observation_map.id = observation.id;
			observations_map.push_back(observation_map);
		}

		dataAssociation(in_range_landmarks, observations_map);

		// Calculating the Particle's Weight
		particles[i].weight = 1.0;
		for(auto &observation : observations_map) {
			auto landmark = in_range_landmarks[observation.id];
			double gauss_norm = 1.0/(2*M_PI*std_landmark[0]*std_landmark[1]);
			auto exponent = (pow(observation.x - landmark.x,2)/(2*std_landmark[0]*std_landmark[0]) + pow(observation.y - landmark.y,2)/(2*std_landmark[1]*std_landmark[1]));
			particle.weight *= gauss_norm*exp(-exponent);
		}
		weights[i] = particle.weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	/*
	 * Set a particles list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */

	vector<Particle> resampled;
	discrete_distribution<> distribution(weights.begin(), weights.end());
	for (auto &particle : particles){
		resampled.push_back(particles[distribution(gen)]);
	}
	particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
