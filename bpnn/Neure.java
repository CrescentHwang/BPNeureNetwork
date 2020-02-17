package pers.crescent.bpNeureNetwork.bpnn;

public abstract class Neure {

    // 神经元的输入值
    private double input;
    // 神经元的输出值
    private double output;
    // 神经元的反向传播输入值

    // 设置该神经元的输入值
    public void setInput(double input) {
        this.input = input;
        this.output = f(input);
    }

    // 获取该神经元的输出值
    public double getOutput() {
        return output;
    }

    // f()为传递函数
    abstract double f(double input);

    public double getInput() {
        return input;
    }
}
