// uniform_real_distribution example
#include <iostream>
#include <chrono>
#include <random>
#include <typeinfo>

int main()
{
  // construct a trivial random generator engine from a time-based seed:
  unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);

  unsigned seed1= std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::default_random_engine generator1 (seed1);

  std::uniform_real_distribution<double> distribution (0.0,100.0);

  std::cout << "some random numbers between 0.0 and 100.0: " << std::endl;
  for (int i=0; i<10; ++i)
  	// std::cout << typeid(generator).name() <<"\n";
	std::cout << distribution(generator) << "\n";
	std::cout << distribution(generator1) << "\n";

  return 0;
}