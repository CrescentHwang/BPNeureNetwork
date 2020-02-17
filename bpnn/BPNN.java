package pers.crescent.bpNeureNetwork.bpnn;

import static java.lang.Math.random;

// BP神经网络
public class BPNN {
    // 默认第一列为类别
    private BPData[] trainData;
    private BPData[] testData;

    // 输入层
    private InputNeure inputN[];
    // 隐含层
    private HiddenNeure hiddenN[];
    // 输出层
    private OutputNeure outputN[];
    // 学习速率
    private double yita;
    private int inputNumber;
    private int hiddenNumber;
    private int outputNumber;
    // 输入层至隐含层的权重
    private double in2HidWeights[][];
    // 隐含层至输出层的权重
    private double hid2OutWeights[][];

    // 输入参数为训练集数据train data，测试集 test data, 学习速率yita，输入层、隐含层、输出层神经元个数
    public BPNN(BPData[] trainData, BPData[] testData, double yita,
                int inputNumber, int hiddenNumber, int outputNumber) {
        this.trainData = trainData;
        this.testData = testData;
        this.inputNumber = inputNumber;
        this.hiddenNumber = hiddenNumber;
        this.outputNumber = outputNumber;
        in2HidWeights = new double[inputNumber+1][hiddenNumber];
        hid2OutWeights = new double[hiddenNumber+1][outputNumber];
        inputN = new InputNeure[inputNumber];
        hiddenN = new HiddenNeure[hiddenNumber];
        outputN = new OutputNeure[outputNumber];
        this.yita = yita;
        init();
    }

    // 初始化神经网络
    private void init() {
        // 初始化神经元
        for(int i=0; i<inputNumber; i++) {
            inputN[i] = new InputNeure();
        }
        for(int i=0; i<hiddenN.length; i++) {
            hiddenN[i] = new HiddenNeure();
        }
        for(int i=0; i<outputN.length; i++) {
            outputN[i] = new OutputNeure();
        }

        // 初始化权重矩阵
        for(int i=0; i<inputNumber; i++) {
            for(int j=0; j<hiddenNumber; j++) {
                in2HidWeights[i][j] = random();
            }
        }
        for(int i=0; i<hiddenNumber; i++) {
            for(int j=0; j<outputNumber; j++) {
                hid2OutWeights[i][j] = random();
            }
        }

        // 初始化阈值
        for(int j=0; j<hiddenNumber; j++) {
            in2HidWeights[inputNumber][j] = random();
        }
        for(int j=0; j<outputNumber; j++) {
            hid2OutWeights[hiddenNumber][j] = random();
        }
    }

    // 单次的向前传播
    private void forward(double[] attributes) {
        // 设置输入层输入
        for(int i=0; i<inputNumber; i++) {
            inputN[i].setInput(attributes[i]);
        }

        // 隐含层
        for(int i=0; i<hiddenN.length; i++) {
            double temp = 0;
            for(int j=0; j<inputNumber; j++) {
                temp += in2HidWeights[j][i] * inputN[j].getOutput();
            }
            hiddenN[i].setInput(temp - in2HidWeights[inputNumber][i]);
        }

        // 输出层
        for (int i = 0; i < outputN.length; i++){
            double temp = 0;
            for(int j=0; j < hiddenN.length; j++) {
                temp += hid2OutWeights[j][i] * hiddenN[j].getOutput();
            }
            outputN[i].setInput(temp - hid2OutWeights[hiddenN.length][i]);
        }
    }

    // 单次的反向传播
    private void backward(double tags[]) {
        double outputNErrorSum[] = new double[hiddenN.length];
        double outputNError[] = new double[outputN.length];
        double hiddenNError[] = new double[hiddenN.length];

        // 输出层误差
        for(int i=0; i<outputN.length; i++) {
            // 计算输出层误差
            double error = tags[i] - outputN[i].getOutput();
//            System.out.println("期望： " + tags[i]);
//            System.out.println("实际： " + outputN[i].getOutput());
//            System.out.println("误差： " + error);
            outputNError[i] = outputN[i].getOutput() * (1 - outputN[i].getOutput()) * error;
//            System.out.println("输出层误差： " + outputNError[i]);
//            System.out.println();
        }

        // 更新权值
        for(int j=0; j<hiddenN.length; j++) {
            for(int k=0; k<outputN.length; k++) {
                // 更新隐含层到输出层权重
                double deltaHid2Out = yita * hiddenN[j].getOutput() * outputNError[k];
                outputNErrorSum[j] += outputNError[k] * hid2OutWeights[j][k];
                hid2OutWeights[j][k] += deltaHid2Out;
            }
        }

        // 隐含层误差
        for(int j=0; j<hiddenN.length; j++) {
            hiddenNError[j] = hiddenN[j].getOutput() * (1 - hiddenN[j].getOutput())
                    * outputNErrorSum[j];
        }
        // 更新权值
        for(int i=0; i<inputNumber; i++) {
            for(int j=0; j<hiddenN.length; j++) {
                double deltaInt2Hid = yita * inputN[i].getOutput() * hiddenNError[j];
                in2HidWeights[i][j] += deltaInt2Hid;
            }
        }
//        System.out.println(deltaInt2Hid);
    }

    // 迭代训练
    public void train_2(int times) {
        for(int i=0; i<times; i++) {
            for(int j=0; j<trainData.length; j++) {
                int attrsLength = trainData[j].getAttributes().length-1;
                double attrs[] = new double[attrsLength];
                double tag = trainData[j].getAttributes()[0];
                System.arraycopy(trainData[j].getAttributes(), 1, attrs,
                                0, attrsLength);
                forward(attrs);
                double tags[] = new double[outputNumber];
                for(int k=0; k<tags.length; k++) {
                    tags[k] = -1;
                }
                tags[(int)tag - 1] = 1;
                backward(tags);
            }
        }

    }

    public void predict_2() {
        train_2(1000);
        for(int j=0; j<testData.length; j++) {
            int attrsLength = testData[j].getAttributes().length-1;
            double attrs[] = new double[attrsLength];
            System.arraycopy(testData[j].getAttributes(), 1, attrs,
                    0, attrsLength);
            double tag = testData[j].getAttributes()[0];
            forward(attrs);
            System.out.println("真实 ：" + tag);
            double max = 0;
            for(int i=0; i<outputN.length; i++) {
//                System.out.println(outputN[i].getOutput());
                if(outputN[i].getOutput() > max) {
                    max = i;
                }
            }
            System.out.println("预测 ：" + max);
            System.out.println();
        }
    }

}




















