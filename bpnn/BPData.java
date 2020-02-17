package pers.crescent.bpNeureNetwork.bpnn;

// BP神经网络的数据对象
public class BPData {

    // 数据的属性集合，默认该数据已全部规范化（即都在0-1范围内）
    double attributes[];

    public BPData() {}

    // 设置数据属性值
    public void setAttributes(double[] attributes) {
        this.attributes = attributes;
    }

    // 获取数据属性值
    public double[] getAttributes() {
        return attributes;
    }

}
