#include<iostream>
#include<cstdio>
#include <fstream>
#include<string>
using namespace std;

int main(int argc, char* argv[])
{
    FILE* fp;
    fp = fopen(argv[1], "r");
    int seq;
    int bitwidth;
    int b_1_3=0;
    int b_4_6=0;
    int b_7_9=0;
    int b_10_12=0;
    int b_13_15=0;
    int b_16_18=0;
    int b_19_21=0;
    int b_22_24=0;
    int b_25_27=0;
    int b_28_30=0;
    int b_else=0;
    while(!feof(fp))
    {
        fscanf(fp, "%d %d\n", &seq, &bitwidth);
        if(bitwidth>=1 && bitwidth<=3)
            b_1_3++;
        else if(bitwidth>=4 && bitwidth<=6)
            b_4_6++;
        else if(bitwidth>=7 && bitwidth<=9)
            b_7_9++;
        else if(bitwidth>=10 && bitwidth<=12)
            b_10_12++;
        else if(bitwidth>=13 && bitwidth<=15)
            b_13_15++;
        else if(bitwidth>=16 && bitwidth<=18)
            b_16_18++;
        else if(bitwidth>=19 && bitwidth<=21)
            b_19_21++;
        else if(bitwidth>=22 && bitwidth<=24)
            b_22_24++;
        else if(bitwidth>=25 && bitwidth<=27)
            b_25_27++;
        else if(bitwidth>=28 && bitwidth<=30)
            b_28_30++;
        else 
            b_else++;
    }

    std::string s = argv[1];
    s += ".new";
    ofstream myfile(s.c_str());
    if (myfile.is_open())
    {
        myfile << "1-3  " << b_1_3 << "\n";
        myfile << "4-6  " << b_4_6 << "\n";
        myfile << "7-9  " << b_7_9 << "\n";
        myfile << "10-12  " << b_10_12 << "\n";
        myfile << "13-15  " << b_13_15 << "\n";
        myfile << "16-18  " << b_16_18 << "\n";
        myfile << "19-21  " << b_19_21 << "\n";
        myfile << "22-24  " << b_22_24 << "\n";
        myfile << "25-27  " << b_25_27 << "\n";
        myfile << "28-30  " << b_28_30 << "\n";
        myfile.close();
    }
    fclose(fp);
}
