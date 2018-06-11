using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace MLnet
{
    class Program
    {
        //第一步：定义数据结构
        //IrisData.txt提供训练数据，同时作为预测运算的输入
        //前四个属性作为输入/特征值用来预测Label,其在训练时被设置

        //data 类型：setosa\versicolor\virginica
        public class IrisData
        {
            [Column("0")]
            public float SepalLength;
            [Column("1")]
            public float SepalWidth;
            [Column("2")]
            public float PetalLength;
            [Column("3")]
            public float PetalWidth;
            [Column("4")]
            [ColumnName("Label")]
            public string Label;
        }

        //预测运算结束后返回IrisPrediction
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }
        static void Main(string[] args)
        {
            // 第二步：创建一个管道载入数据
            var pipeline = new LearningPipeline();

            string dataPath = "Irisdata.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            //第三步：
            pipeline.Add(new Dictionarizer("Label"));

            //将所有特征添加进向量
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            //第四步：添加学习者
            //向管道中添加机器学习算法
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            //第五步：训练数据
            var model = pipeline.Train<IrisData, IrisPrediction>();
            //输入需要预测的模型

            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f,
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");

            Console.ReadKey();
        }

    }
}

