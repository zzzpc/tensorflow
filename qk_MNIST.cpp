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

int sum = 0;
char NDb[M][6272] = {};
char tempstr[M];
char tempstr1[M];
char tempstr2[M];
char  cc[256][10] =
{ { "00000000" }, { "00000001" }, { "00000010" }, { "00000011" }, { "00000100" }, { "00000101" }, { "00000110" }, { "00000111" }, { "00001000" }, { "00001001" }, { "00001010" }
, { "00001011" }, { "00001100" }, { "00001101" }, { "00001110" }, { "00001111" }, { "00010000" }, { "00010001" }, { "00010010" }, { "00010011" }, { "00010100" }, { "00010101" }
, { "00010110" }, { "00010111" }, { "00011000" }, { "00011001" }, { "00011010" }, { "00011011" }, { "00011100" }, { "00011101" }, { "00011110" }, { "00011111" }, { "00100000" }
, { "00100001" }, { "00100010" }, { "00100011" }, { "00100100" }, { "00100101" }, { "00100110" }, { "00100111" }, { "00101000" }, { "00101001" }, { "00101010" }, { "00101011" }
, { "00101100" }, { "00101101" }, { "00101110" }, { "00101111" }, { "00110000" }, { "00110001" }, { "00110010" }, { "00110011" }, { "00110100" }, { "00110101" }, { "00110110" }
, { "00110111" }, { "00111000" }, { "00111001" }, { "00111010" }, { "00111011" }, { "00111100" }, { "00111101" }, { "00111110" }, { "00111111" }, { "01000000" }, { "01000001" }
, { "01000010" }, { "01000011" }, { "01000100" }, { "01000101" }, { "01000110" }, { "01000111" }, { "01001000" }, { "01001001" }, { "01001010" }, { "01001011" }, { "01001100" }
, { "01001101" }, { "01001110" }, { "01001111" }, { "01010000" }, { "01010001" }, { "01010010" }, { "01010011" }, { "01010100" }, { "01010101" }, { "01010110" }, { "01010111" }
, { "01011000" }, { "01011001" }, { "01011010" }, { "01011011" }, { "01011100" }, { "01011101" }, { "01011110" }, { "01011111" }, { "01100000" }, { "01100001" }, { "01100010" }
, { "01100011" }, { "01100100" }, { "01100101" }, { "01100110" }, { "01100111" }, { "01101000" }, { "01101001" }, { "01101010" }, { "01101011" }, { "01101100" }, { "01101101" }
, { "01101110" }, { "01101111" }, { "01110000" }, { "01110001" }, { "01110010" }, { "01110011" }, { "01110100" }, { "01110101" }, { "01110110" }, { "01110111" }, { "01111000" }
, { "01111001" }, { "01111010" }, { "01111011" }, { "01111100" }, { "01111101" }, { "01111110" }, { "01111111" }, { "10000000" }, { "10000001" }, { "10000010" }, { "10000011" }
, { "10000100" }, { "10000101" }, { "10000110" }, { "10000111" }, { "10001000" }, { "10001001" }, { "10001010" }, { "10001011" }, { "10001100" }, { "10001101" }, { "10001110" }
, { "10001111" }, { "10010000" }, { "10010001" }, { "10010010" }, { "10010011" }, { "10010100" }, { "10010101" }, { "10010110" }, { "10010111" }, { "10011000" }, { "10011001" }
, { "10011010" }, { "10011011" }, { "10011100" }, { "10011101" }, { "10011110" }, { "10011111" }, { "10100000" }, { "10100001" }, { "10100010" }, { "10100011" }, { "10100100" }
, { "10100101" }, { "10100110" }, { "10100111" }, { "10101000" }, { "10101001" }, { "10101010" }, { "10101011" }, { "10101100" }, { "10101101" }, { "10101110" }, { "10101111" }
, { "10110000" }, { "10110001" }, { "10110010" }, { "10110011" }, { "10110100" }, { "10110101" }, { "10110110" }, { "10110111" }, { "10111000" }, { "10111001" }, { "10111010" }
, { "10111011" }, { "10111100" }, { "10111101" }, { "10111110" }, { "10111111" }, { "11000000" }, { "11000001" }, { "11000010" }, { "11000011" }, { "11000100" }, { "11000101" }
, { "11000110" }, { "11000111" }, { "11001000" }, { "11001001" }, { "11001010" }, { "11001011" }, { "11001100" }, { "11001101" }, { "11001110" }, { "11001111" }, { "11010000" }
, { "11010001" }, { "11010010" }, { "11010011" }, { "11010100" }, { "11010101" }, { "11010110" }, { "11010111" }, { "11011000" }, { "11011001" }, { "11011010" }, { "11011011" }
, { "11011100" }, { "11011101" }, { "11011110" }, { "11011111" }, { "11100000" }, { "11100001" }, { "11100010" }, { "11100011" }, { "11100100" }, { "11100101" }, { "11100110" }
, { "11100111" }, { "11101000" }, { "11101001" }, { "11101010" }, { "11101011" }, { "11101100" }, { "11101101" }, { "11101110" }, { "11101111" }, { "11110000" }, { "11110001" }
, { "11110010" }, { "11110011" }, { "11110100" }, { "11110101" }, { "11110110" }, { "11110111" }, { "11111000" }, { "11111001" }, { "11111010" }, { "11111011" }, { "11111100" }
, { "11111101" }, { "11111110" }, { "11111111" } };
char s[M];
int m;
int cn;
double p[4];
double q[8];
double r = 6.5;
typedef struct{
	int p[3];				//三个位 
	char c[4];
}Ent;
Ent NDB[N];
int Pos[6272][2] = { 0 };
double Gen[10000][784];
double posbility[8][2] = { 0 };

double  diff(int i){                  //计算与原串不同的概率
	double Ndiff = 0;
	double Nsame = 0;
	for (int j = 1; j <= 3; j++){
		Ndiff += j*p[j] * q[i];
	}
	for (int j = 1; j <= 3; j++){
		Nsame += ((3 - j)*p[j]) / 8;
	}

	double Pdiff = 0;
	Pdiff = Ndiff / (Ndiff + Nsame);
	return Pdiff;

}
void init()
{
	m = 6272;
	r = 6.5;
	p[0] = 0;
	p[1] = 1;
	p[2] = 0;
	p[3] = 1 - p[1] - p[2];
	cn = int(m*r + 0.5);


	q[0] = 0.3;
	q[1] = 0.1;
	q[2] = 0.1;
	q[3] = 0.1;
	q[4] = 0.1;
	q[5] = 0.1;
	q[6] = 0.1;
	q[7] = 0.1;


	for (int i = 0; i < 8; i++){
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

	if (l < q[7]) return 7;
	else if (l < q[7] + q[6]) return 6;
	else if (l < q[7] + q[6] + q[5])return 5;
	else if (l < q[7] + q[6] + q[5] + q[4])return 4;
	else if (l < q[7] + q[6] + q[5] + q[4] + q[3])return 3;
	else if (l < q[7] + q[6] + q[5] + q[4] + q[3] + q[2])return 2;
	else if (l < q[7] + q[6] + q[5] + q[4] + q[3] + q[2] + q[1])return 1;
	else return 0;

}

void  geneRandombit(Ent &v, int l){

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
	double u;
	n = int(m*r + 0.5);


	//四舍五入 
	cn = 0;
	//clock_t start,end; // typedef long clock_t
	//    start = clock();
	do
	{
		//取3个随机位 
		t = rand1();				//生成0~1之间的小数 
		if (t<p[1])                 //生成类型一
		{
			int diff1 = 0;
			int same1 = 0;
			int same2 = 0;
			int attr1 = 0;
			u = rand1();
			diff1 = generateRandomNumbers(u);  //通过属性q决定属性的哪一位与原始位不同
			attr1 = rand1(m / 8);
			v.p[0] = diff1 + attr1 * 8 ;      //生成的反转位
			v.c[0] = '1' + '0' - s[v.p[0]];

			same1 = rand1(8);
			while (same1 == diff1){
				same1 = rand1(8);
			}
			v.p[1] = same1 + attr1 * 8 ;
			v.c[1] = s[v.p[1]];

			same2 = rand1(8);
			while (same2 == diff1 || same2 == same1){
				same2 = rand1(8);
			}
			v.p[2] = same2 + attr1 * 8;
			v.c[2] = s[v.p[2]];


		}
		else if (t<p[1] + p[2])
		{
			int diff1 = 0;
			int diff2 = 0;
			int same1 = 0;
			int attr = 0;

			diff1 = generateRandomNumbers(rand1());  //通过属性q决定属性的哪一位与原始位不同
			diff2 = generateRandomNumbers(rand1());

			attr = rand1(m / 8);
			v.p[0] = diff1 + attr * 8 ;      //生成的反转位
			v.c[0] = '1' + '0' - s[v.p[0]];
			while (diff2 == diff1){
				diff2 = generateRandomNumbers(rand1());
			}

			v.p[1] = diff2 + attr * 8;
			v.c[1] = '1' + '0' - s[v.p[1]];

			same1 = rand1(8);
			while (same1 == diff1 || same1 == diff2){
				same1 = rand1(8);
			}
			v.p[2] = same1 + attr * 8;
			v.c[2] = s[v.p[2]];
		}
		else
		{
			int diff1 = 0;
			int diff2 = 0;
			int diff3 = 0;
			int attr = 0;
			diff1 = generateRandomNumbers(rand1());  //通过属性q决定属性的哪一位与原始位不同
			attr = rand1(m / 8);
			v.p[0] = diff1 + attr * 8;      //生成的反转位
			v.c[0] = '1' + '0' - s[v.p[0]];
			diff2 = generateRandomNumbers(rand1());
			while (diff2 == diff1){
				diff2 = generateRandomNumbers(rand1());
			}
			v.p[1] = diff2 + attr * 8;      //生成的反转位
			v.c[1] = '1' + '0' - s[v.p[1]];
			diff3 = generateRandomNumbers(rand1());
			while (diff3 == diff1 || diff3 == diff2){
				diff3 = generateRandomNumbers(rand1());
			}
			v.p[2] = diff3 + attr * 8;      //生成的反转位
			v.c[2] = '1' + '0' - s[v.p[2]];

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



int num_01(){

	for (int i = 0; i < cn; i++){
		for (int j = 0; j < 3; j++){
			if (NDB[i].c[j] == '0')    Pos[NDB[i].p[j]][0]++;
			else Pos[NDB[i].p[j]][1]++;
		}
	}
	

	return 0;
}

double  Pr(int index, int num){       //传入的值为原始穿中第i位的索引值(可以认为第i个属性值的第一位索引值    num为原串在该位的值为0或者1)
	int n0 = 0;
	int n1 = 1;
	double pr1 = 0;
	double pr0 = 0;
	int tmp = index % 8;
	double pdiff = diff(tmp);    //posbility[tmp][0];
	double psame = 1 - pdiff;     //posbility[tmp][1];
	if (num == 1){
		pr1 = (pow(pdiff, Pos[index][0])*pow(psame, Pos[index][1])) / (pow(pdiff, Pos[index][1])*pow(psame, Pos[index][0]) + pow(pdiff, Pos[index][0])*pow(psame, Pos[index][1]));
		return pr1;
	}
	else{
		pr0 = (pow(pdiff, Pos[index][1])*pow(psame, Pos[index][0])) / (pow(pdiff, Pos[index][1])*pow(psame, Pos[index][0]) + pow(pdiff, Pos[index][0])*pow(psame, Pos[index][1]));
		return pr0;
	}


}
double calculate(int index, int real){        //计算图片每个像素为0-255的概率
	double sum = 1;
	double tmp;
	for (int i = 0; i < 8; i++){
		tmp = Pr(index + i, cc[real][i] - '0');
		sum = sum* tmp;
	}
	sum *= real;
	return sum;
}
double  E(int index){      //计算期望值   
	double tmp = 0;
	for (int i = 0; i < 256; i++){
		tmp += calculate(index, i);
	}
	//tmp = tmp / 256;
	return tmp;
}
void readfile(int count){

	int count2 = 0;
	for (int i = 0; i < 6272; i++){
		if (i % 8 == 0)		Gen[count][count2++] = E(i);     //每隔8位传入原始01串的索引值生成8位01串对应0-255每位值的期望值 count2代表最后生成矩阵的元素个数
	}

	return;
}
int main(){

	init();
	ifstream myfile;
	string tmp;
	myfile.open("H:\\cc_test.txt");
	int count = 0;
	while (!myfile.eof())   //按行读取,遇到换行符结束
	{
		myfile >> tmp;
		if (myfile.peek() == EOF)  break;
		(strcpy(NDb[count++], tmp.c_str()));
		if (count == 100) break;


	}
	myfile.close();
	cout << count << "张图片的01串已经生成" << endl;

	time_t  start = time(NULL);

	for (int i = 0; i < count; i++){
		f(NDb[i]); //生成每张图片所对应的负数据库记录。
		num_01();    //计算负数据库中所有记录条数中相对于原串中为位分别为0 或者1的个数
		readfile(i);  //生成每张求完期望之后的图片
		for (int i = 0; i < 6272; i++){
			for (int j = 0; j < 2; j++){
				Pos[i][j] = 0;
			}
		}
	}
	time_t  end = time(NULL);
	cout << "负数据库生成以及重构执行持续时间：" << difftime(end, start) << "秒" << endl;
	ofstream   outfile;
	outfile.open("H:\\k.txt", ios::app);
	for (int i = 0; i < count; i++){
		for (int j = 0; j <784; j++){
			outfile << Gen[i][j] << " ";
		}
		outfile << endl;

	}
	outfile.close();

}
