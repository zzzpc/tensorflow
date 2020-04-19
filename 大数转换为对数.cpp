#define _CRT_SECURE_NO_WARNINGS
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include <iostream>
#include<cstring>
#include<fstream>
#include<sstream>
#include <algorithm>
using namespace std;


double lnchoose(int n, int m)
{

	if (m > n)

	{

		return 0;

	}
	if (m < n / 2.0)
	{
		m = n - m;
	}

	double s1 = 0;
	for (int i = m + 1; i <= n; i++)
	{
		s1 += log((double)i);
	}

	double s2 = 0;
	int ub = n - m;
	for (int i = 2; i <= ub; i++)
	{
		s2 += log((double)i);
	}

	return s1 - s2;
}

double choose(int n, int m)
{

	if (m > n)

	{

		return 0;

	}
	return (lnchoose(n, m)/log(2));
}

int main(){

	double JC;
	JC = choose(1536, 307);
	cout << JC;
}
