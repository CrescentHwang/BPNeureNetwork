package pers.crescent.bpNeureNetwork.bpnn;

import static java.lang.Math.exp;

public class OutputNeure extends Neure{

    // 此处使用sigmoid()函数作为激活函数
    @Override
    double f(double input) {
        return (1 / (1 + exp(-input)));
    }
}
