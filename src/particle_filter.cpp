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


double ParticleFilter::addNoise(double mean, double std) {
	normal_distribution<double> distr(mean, std);
  return distr(gen);
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 10;

	for (unsigned int i = 0; i < num_particles; i++) {
	  Particle p;
	  p.id = i;
	  p.x = addNoise(x, std[0]);
	  p.y = addNoise(y, std[1]);;
	  p.theta = addNoise(theta, std[2]);;
	  p.weight = 1.0;
	  particles.push_back(p);
	  weights.push_back(p.weight);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


  //predicting new position
	double dist = velocity * delta_t;
	double delta_theta = yaw_rate * delta_t;
	double turn = fabs(yaw_rate) < 0.001 ? 0.0 : velocity / yaw_rate;

	for (Particle& p : particles) {
	  double theta2 = p.theta + delta_theta;
 
	  if (turn == 0.0) {
	    p.x += dist * cos(p.theta);
	    p.y += dist * sin(p.theta);
	  }
	  else {
	    p.x += turn * (sin(theta2) - sin(p.theta));
	    p.y += turn * (cos(p.theta) - cos(theta2));
	    p.theta = theta2;
	  }
    p.x = addNoise(p.x, std_pos[0]);
    p.y = addNoise(p.y, std_pos[1]);
    p.theta = addNoise(p.theta, std_pos[2]);
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	for (unsigned int i = 0; i < particles.size(); i++) {
	  Particle& p = particles[i];
   	std::vector<int> associations; 
   	std::vector<double> sense_x; 
   	std::vector<double> sense_y;
   	
   	//building the mapLandmarks within the proximity
   	vector<LandmarkObs> predicted;
  	for (Map::single_landmark_s& ml : map_landmarks.landmark_list){
      double d = dist(p.x, p.y, ml.x_f, ml.y_f);
      if (d < sensor_range){
        LandmarkObs l;
        l.x = ml.x_f;
        l.y = ml.y_f;
        l.id = ml.id_i;
        predicted.push_back(l);
      }    	  
  	}
   	
   	if (predicted.size() > 0){
     	p.weight = 1.0;
     	
    	for (LandmarkObs& o : observations){
    	  //converting observations into the MAP's coordinate system
    	  LandmarkObs mo;
    	  mo.x = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
    	  mo.y = p.y + o.x * sin(p.theta) + o.y * cos(p.theta);
    	  mo.id = o.id;
    	  
    	  //finding the closest predicted map landmark
        double distance;
        int index = -1;
      	for (unsigned int j = 0; j < predicted.size(); j++){
          double d = dist(mo.x, mo.y, predicted[j].x, predicted[j].y);
          if (distance > d || index == -1){
            distance = d;
            index = j;
          }    	  
      	}
      	
      	//calculating weight
    	  LandmarkObs l = predicted[index];

    	  p.weight *= 0.5 / M_PI / std_landmark[0] / std_landmark[1] *
    	    exp(
    	      - pow(mo.x - l.x, 2) / 2 / pow(std_landmark[0], 2)
    	      - pow(mo.y - l.y, 2) / 2 / pow(std_landmark[1], 2)
    	    );

cout << p.weight << endl;
    	  //building associations
    	  associations.push_back(l.id);
    	  sense_x.push_back(mo.x);
    	  sense_y.push_back(mo.y);
	    }
	  } else {
	    p.weight = 0.0;
	  }
	  weights[i] = p.weight;

    //setting associations
    particles[i] = SetAssociations(p, associations, sense_x, sense_y);
	  
	}

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d (weights.begin(), weights.end());

	std::vector<Particle> ps2;
  for (unsigned int i = 0; i < num_particles; i++){
    Particle p = particles[d(gen)];
    ps2.push_back(p);
  }
  particles = ps2;
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
