package demo;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;


public class Regression {

    final static long SEED = 1234L;
    final static int trainSize = 400;
    final static int batchSize = 4;


    public static void main(String[] args) throws Exception {
        run(14,1);
    }

    private static void run(int csvSize,int outSize) throws IOException {
        List<DataSet> dataSets = getDatas(new File("D:\\housing.csv"),outSize);


        DataSet allData = DataSet.merge(dataSets);
        allData.shuffle(SEED);

        SplitTestAndTrain split = allData.splitTestAndTrain(trainSize);

        DataSet dsTrain = split.getTrain();
        DataSet dsTest = split.getTest();

        DataSetIterator trainIter = new ListDataSetIterator(dsTrain.asList() , batchSize);
        DataSetIterator testIter = new ListDataSetIterator(dsTest.asList() , batchSize);

        DataNormalization scaler = new NormalizerMinMaxScaler(0,1);

        scaler.fit(trainIter);
        scaler.fit(testIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);


        MultiLayerNetwork network = createModel(csvSize-outSize,outSize);

        for( int i = 0; i < 200; ++i ){
            network.fit(trainIter);
            trainIter.reset();
        }

        RegressionEvaluation eval = network.evaluateRegression(testIter);
        System.out.println(eval.stats());
        testIter.reset();

        testIter.reset();

        System.out.println(network.output(testIter));
        testIter.reset();


        boolean saveUpdater = true;
        ModelSerializer.writeModel(network, new File("D:\\Regression_housingx.mod"), saveUpdater,scaler);

    }


    public static MultiLayerNetwork createModel(int inputSize,int outsize) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(123456)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .list()
                .layer(0, new DenseLayer.Builder().activation(Activation.LEAKYRELU)
                        .nIn(inputSize).nOut(10).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.IDENTITY)
                        .nIn(10).nOut(outsize).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        net.setListeners(new ScoreIterationListener(100));
        net.setListeners(new StatsListener(statsStorage));

        return net;
    }

    public static List<DataSet> getDatas(File file,int outSize) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line = null;

        List<DataSet> totalDataSetList = new LinkedList<DataSet>();
        while ((line = br.readLine()) != null) {
            String[] token = line.split(",");
            double[] featureArray = new double[token.length - outSize];
            double[] labelArray = new double[outSize];
            for (int i = 0; i < token.length - (outSize); ++i) {
                featureArray[i] = Double.parseDouble(token[i]);
            }

            for (int i = 0; i < outSize; i++) {
                labelArray[i] = Double.parseDouble(token[token.length - (i+1)]);

            }

            INDArray featureNDArray = Nd4j.create(featureArray);
            INDArray labelNDArray = Nd4j.create(labelArray);
            totalDataSetList.add(new DataSet(featureNDArray, labelNDArray));
        }
        br.close();
        return totalDataSetList;


    }

}
