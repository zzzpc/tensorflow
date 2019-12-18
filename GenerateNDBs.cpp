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
#define N 600000
#define M 10000
#define RN 500
#define RM 20

int sum = 0;

char tempstr[M];
char tempstr1[M];
char tempstr2[M];

char s[M];
int m;
int cn;
double p[4];
double r = 6.5;
typedef struct{
	int p[3];				//三个位 
	char c[4];
}Ent;
Ent NDB[N];
void init()
{   
	m = 6272;
	r = 6.5;
	p[0] = 0;
	p[1] = 0.80;
	p[2] = 0.14;
	p[3] = 1 - p[1] - p[2];
	cn = int(m*r + 0.5);
}

int rand1(int n)
{
	//	return rand()%n;
	return 0 + (int)n * rand() / (RAND_MAX + 1);
}

double rand1()
{
	return (double)rand() / (RAND_MAX + 1.0);
}

void generateRandomNumbers(Ent &v, int l)	//取了三个随机位 
{
	int temp;
	v.p[0] = rand1(l);
	temp = rand1(l);

	while (temp == v.p[0])
	{
		temp = rand1(l);
	}
	v.p[1] = temp;
	temp = rand1(l);
	while ((temp == v.p[0]) || (temp == v.p[1]))
	{
		temp = rand1(l);
	}
	v.p[2] = temp;
}
void addToNDB(Ent x)		//负数据库确定位赋值 
{
	int i;
	for (i = 0; i<3; i++)
	{
		NDB[cn].p[i] = x.p[i];
		NDB[cn].c[i] = x.c[i];
	}
	cn++;
}
void f(char s[])		//生成s[]的负数据库	
{
	Ent v;
	int i, n;
	double t;
	int u;
	n = int(m*r + 0.5);	
	
	//四舍五入 
	cn = 0;
	//clock_t start,end; // typedef long clock_t
	//    start = clock();
	do
	{
		generateRandomNumbers(v, m);			//取3个随机位 
		t = rand1();				//生成0~1之间的小数 
		if (t<p[1])
		{
			u = rand1(3);
			for (i = 0; i<3; i++)
			{
				v.c[i] = s[v.p[i]];
			}
			v.c[u] = '1' + '0' - s[v.p[u]]; 	//随机取一位取反 
		}
		else if (t<p[1] + p[2])
		{
			u = rand1(3);
			for (i = 0; i < 3; i++)
			{  
				
				v.c[i] = '1' + '0' - s[v.p[i]];		//三位取反 
			}
			v.c[u] = s[v.p[u]];					//选一位复原 
		}
		else
		{
			for (i = 0; i<3; i++)
			{
				v.c[i] = '1' + '0' - s[v.p[i]];		//三位取反 
			}
		}
		addToNDB(v);		//负数据库确定位赋值 
	} while (cn<n);
	
}

void printNDB()				//输出确定位和确定位的值 
{    
	ofstream  outfile;
	int count = 0;
	int i, j;
	char index[100] = { 0 };
			sprintf(index, "G:\\python\\zpc_out.txt");
			outfile.open(index,ios::app);
			for (i = 0; i < cn; i++)
			{
				for (j = 0; j < 3; j++)
				{
					outfile << NDB[i].p[j] << " " << NDB[i].c[j] << "  ";
					count++;
					//printf("%d:%c  ", NDB[i].p[j], NDB[i].c[j]);
				}
				outfile << endl;
			}
			outfile.close();
}

int main(){
	
		ifstream myfile;
		string tmp;
		char index[100] = { 0 };
		char NDb[N];
		int count = 0;
		init();
		time_t  start = time(NULL);
			sprintf(index, "G:\\python\\zpc.txt");
			myfile.open(index);
			while (getline(myfile, tmp))   //按行读取,遇到换行符结束
			{
				count++;
				f(strcpy(NDb, tmp.c_str()));
				printNDB();
				//cout<<tmp << endl;
			} 
				    
		time_t  end = time(NULL);
		cout << "算法执行持续时间：" << difftime(end, start) << "秒" << count<<endl;

	
}
