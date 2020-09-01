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
#define M 30000
#define RN 500
#define RM 20
#define LEN 36


double traindata[M][10];
double testdata[N][10];
char binTraindata[M][LEN];
char binTestdata[N][LEN];
int trainY[M];
int testY[N];
int trainN = 483;
int testN = 205;
int classN = 2;
const int L = 4;


int sum = 0;
char NDb[M][36] = {};

char  cc[16][10] =
{ { "0000" }, { "0001" }, { "0010" }, { "0011" }, { "0100" }, { "0101" }, { "0110" }, { "0111" }, { "1000" }, { "1001" },
{ "1010" }, { "1011" }, { "1100" }, { "1101" }, { "1110" }, { "1111" }, };
char s[M];
int m;
int cn;
double p[4];
double q[4];
double r;
typedef struct {
	int p[3];				//三个位 
	char c[4];
}Ent;
Ent NDB[N];
int Pos[36][2] = { 0 };
double Gen[1000][36];
double posbility[4][2] = { 0 };


double  diff(int i) {                  //计算每一位与原串不同的概率
	double Ndiff = 0;
	double Nsame = 0;
	for (int j = 1; j <= 3; j++) {
		Ndiff += j * p[j] * q[i];
	}
	for (int j = 1; j <= 3; j++) {
		Nsame += ((3 - j) * p[j]) / 4;
	}

	double Pdiff = 0;
	Pdiff = Ndiff / (Ndiff + Nsame);
	return Pdiff;

}
void init()
{
	m = 36;
	r = 6.5;
	p[0] = 0;
	p[1] = 0.7;
	p[2] = 0.24;
	p[3] = 1 - p[1] - p[2];
	cn = int(m * r + 0.5);


	q[0] = 0.1;
	q[1] = 0.1;
	q[2] = 0.1;
	q[3] = 0.70;



	for (int i = 0; i < 4; i++) {
		posbility[i][0] = diff(i);
		posbility[i][1] = 1 - posbility[i][0];
	}
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

int  generateRandomNumbers(double l)	//取了三个随机位 
{

	if (l < q[3]) return 3;
	else if (l < q[3] + q[2]) return 2;
	else if (l < q[3] + q[2] + q[1])return 1;
	else return 0;

}

void  geneRandombit(Ent& v, int l) {

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
	for (i = 0; i < 3; i++)
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
	double u;
	n = int(m * r + 0.5);


	//四舍五入 
	cn = 0;

	do
	{
		//取3个随机位 
		t = rand1();				//生成0~1之间的小数 
		if (t < p[1])                 //生成类型一
		{
			int diff1 = 0;
			int same1 = 0;
			int same2 = 0;
			int attr1 = 0;
			u = rand1();
			diff1 = generateRandomNumbers(u);  //通过属性q决定属性的哪一位与原始位不同
			attr1 = rand1(m / 4);
			v.p[0] = diff1 + attr1 * 4;      //生成的反转位
			v.c[0] = '1' + '0' - s[v.p[0]];

			same1 = rand1(4);
			while (same1 == diff1) {
				same1 = rand1(4);
			}
			v.p[1] = same1 + attr1 * 4;
			v.c[1] = s[v.p[1]];

			same2 = rand1(4);
			while (same2 == diff1 || same2 == same1) {
				same2 = rand1(4);
			}
			v.p[2] = same2 + attr1 * 4;
			v.c[2] = s[v.p[2]];


		}
		else if (t < p[1] + p[2])
		{
			int diff1 = 0;
			int diff2 = 0;
			int same1 = 0;
			int attr = 0;

			diff1 = generateRandomNumbers(rand1());  //通过属性q决定属性的哪一位与原始位不同
			diff2 = generateRandomNumbers(rand1());

			attr = rand1(m / 4);
			v.p[0] = diff1 + attr * 4;      //生成的反转位
			v.c[0] = '1' + '0' - s[v.p[0]];
			while (diff2 == diff1) {
				diff2 = generateRandomNumbers(rand1());
			}

			v.p[1] = diff2 + attr * 4;
			v.c[1] = '1' + '0' - s[v.p[1]];

			same1 = rand1(4);
			while (same1 == diff1 || same1 == diff2) {
				same1 = rand1(4);
			}
			v.p[2] = same1 + attr * 4;
			v.c[2] = s[v.p[2]];
		}
		else
		{
			int diff1 = 0;
			int diff2 = 0;
			int diff3 = 0;
			int attr = 0;
			diff1 = generateRandomNumbers(rand1());  //通过属性q决定属性的哪一位与原始位不同
			attr = rand1(m / 4);
			v.p[0] = diff1 + attr * 4;      //生成的反转位
			v.c[0] = '1' + '0' - s[v.p[0]];
			diff2 = generateRandomNumbers(rand1());
			while (diff2 == diff1) {
				diff2 = generateRandomNumbers(rand1());
			}
			v.p[1] = diff2 + attr * 4;      //生成的反转位
			v.c[1] = '1' + '0' - s[v.p[1]];
			diff3 = generateRandomNumbers(rand1());
			while (diff3 == diff1 || diff3 == diff2) {
				diff3 = generateRandomNumbers(rand1());
			}
			v.p[2] = diff3 + attr * 4;      //生成的反转位
			v.c[2] = '1' + '0' - s[v.p[2]];

		}
		addToNDB(v);		//负数据库确定位赋值 
	} while (cn < n);

}

void printNDB()				//输出确定位和确定位的值 
{
	ofstream  outfile;
	int count = 0;
	int i, j;
	char index[100] = { 0 };

	for (i = 0; i < cn; i++)
	{
		for (j = 0; j < 3; j++)
		{
			outfile << NDB[i].p[j] << " " << NDB[i].c[j] << "  ";

		}
		outfile << endl;
	}
	outfile.close();
}



int num_01() {
	for (int i = 0; i < cn; i++) {
		for (int j = 0; j < 3; j++) {
			if (NDB[i].c[j] == '0')    Pos[NDB[i].p[j]][0]++;
			else Pos[NDB[i].p[j]][1]++;
		}
	}


	return 0;
}

double  Pr(int index, int num) {       //传入的值为原始穿中第i位的索引值(可以认为第i个属性值的第一位索引值    num为原串在该位的值为0或者1)
	double pr1 = 0;
	double pr0 = 0;
	int tmp = index % 4;
	double pdiff = posbility[tmp][0];
	double psame = posbility[tmp][1];
	if (num == 1) {
		pr1 = (pow(pdiff, Pos[index][0]) * pow(psame, Pos[index][1])) / (pow(pdiff, Pos[index][1]) * pow(psame, Pos[index][0]) + pow(pdiff, Pos[index][0]) * pow(psame, Pos[index][1]));
		return pr1;
	}
	else {
		pr0 = (pow(pdiff, Pos[index][1]) * pow(psame, Pos[index][0])) / (pow(pdiff, Pos[index][1]) * pow(psame, Pos[index][0]) + pow(pdiff, Pos[index][0]) * pow(psame, Pos[index][1]));
		return pr0;
	}


}
double calculate(int index, int real) {        //计算图片每个像素为0-255的概率
	double sum = 1;
	double tmp;
	for (int i = 0; i < 4; i++) {
		tmp = Pr(index + i, cc[real][i] - '0');
		sum = sum * tmp;
	}
	sum *= real;
	return sum;
}
double  E(int index) {      //计算期望值   
	double tmp = 0;
	for (int i = 0; i < 16; i++) {
		tmp += calculate(index, i);
	}
	//tmp = tmp / 256;
	return tmp;
}
void readfile(int count) {

	int count2 = 0;
	for (int i = 0; i < 36; i++) {
		if (i % 4 == 0)		Gen[count][count2++] = E(i);     //每隔8位传入原始01串的索引值生成8位01串对应0-255每位值的期望值 count2代表最后生成矩阵的元素个数
	}

	return;
}

void readData()
{
	ifstream inFile;
	int L = 9;
	inFile.open("train.txt", ios::in);
	char tc;
	for (int i = 0; i < trainN; i++)
	{
		for (int j = 0; j < L; j++)
		{
			inFile >> traindata[i][j] >> tc;

		}
		inFile >> trainY[i];
	}
	inFile.close();
	inFile.open("test.txt", ios::in);
	for (int i = 0; i < testN; i++)
	{
		for (int j = 0; j < L; j++)
		{
			inFile >> testdata[i][j] >> tc;
		}
		inFile >> testY[i];
	}
	inFile.close();
}

void getBinData()
{
	for (int i = 0; i < trainN; i++)
	{
		int d = 0;
		for (int j = 0; j < 9; j++)
		{
			int v = traindata[i][j];
			//cout << v << " ";
			for (int k = 0; k < L; k++)
			{
				binTraindata[i][d + L - 1 - k] = v % 2 + '0';
				v = v / 2;
			}
			d = d + L;
		}
	}
	for (int i = 0; i < testN; i++)
	{
		int d = 0;
		for (int j = 0; j < 9; j++)
		{
			int v = testdata[i][j];
			for (int k = 0; k < L; k++)
			{
				binTestdata[i][d + L - 1 - k] = v % 2 + '0';
				v = v / 2;
			}
			d = d + L;
		}
	}
}
int main() {


	init();
	for (int i = 0; i < 4; i++) {
		cout << posbility[i][0] << " " << posbility[i][1] << " ";
	}
	readData();
	getBinData();
	string fileout[2] = { { "010_train.txt" },{ "050_test.txt" } };
	
		for (int i = 0; i <478; i++) {
			f(binTraindata[i]); //生成每张图片所对应的负数据库记录。
			num_01();    //计算负数据库中所有记录条数中相对于原串中为位分别为0 或者1的个数
			readfile(i);  //生成每张求完期望之后的图片
			for (int j = 0; j < 36; j++) {
				for (int k = 0; k < 2; k++) {
					Pos[j][k] = 0;
				}
			}
	
		}
		time_t  end = time(NULL);

		ofstream   outfile;
		outfile.open(fileout[0], ios::app);
		for (int i = 0; i < 478; i++) {
			for (int j = 0; j < 9; j++) {
				outfile << Gen[i][j] << ",";
			}
			outfile << trainY[i] << endl;


		}
		outfile.close();


		

	
}
