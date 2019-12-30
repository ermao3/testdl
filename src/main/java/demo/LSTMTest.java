package demo;

import com.google.gson.Gson;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Created by DongBin on 2019/1/25.
 */
public class LSTMTest {
    static String DATA_PATH = "F:/data";

    private static final Logger log = LoggerFactory.getLogger(LSTMTest.class);

    static int NB_TRAIN_EXAMPLES = 2000;// number of training examples
    static int NB_TEST_EXAMPLES = 800; // number of testing examples

    public static void main(String[] args) throws Exception {
       // buildMod();


        test1();


        //test2();

        return;
    }

    private static void test2() throws IOException, InterruptedException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("lstm.mod");

        DataNormalization normalizer =  ModelSerializer.restoreNormalizerFromFile(new File("lstm.mod"));          //Note that we are using the exact same normalization process as the training data
        String path = FilenameUtils.concat(DATA_PATH, "physionet2012/");


        String featureBaseDir = FilenameUtils.concat(path, "sequence"); // set feature directory
        String mortalityBaseDir = FilenameUtils.concat(path, "mortality"); // set label directory


        CSVSequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
        testFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 3500, 3999));

        CSVSequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", 3500, 3999));
        testLabels.sequenceRecord();

        SequenceRecordReaderDataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
            1, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        testData.setPreProcessor(normalizer);

        //Evaluate on the test set:
        Evaluation evaluation = model.evaluate(testData);
        log.info(evaluation.stats());
        testData.setPreProcessor(normalizer);




        /*while (testData.hasNext()){
            DataSet data =  testData.next();
            INDArray in =  testData.next().getFeatures();
            INDArray indArray  =  model.output(in);
            System.out.println("==========================================================");
            Gson gson = new Gson();
            System.out.print(indArray.getColumn(0).getColumn(indArray.getColumn(0).length()-1)+"   ");
            System.out.println(indArray.getColumn(1).getColumn(indArray.getColumn(1).length()-1));

            System.out.print(data.getLabels().getColumn(0).getColumn(data.getLabels().getColumn(0).length()-1).toString()+"   ");
            System.out.println(data.getLabels().getColumn(1).getColumn(data.getLabels().getColumn(1).length()-1).toString());

        }*/
    }

    private static void test1() throws IOException, InterruptedException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("lstm.mod");

        DataNormalization normalizer =  ModelSerializer.restoreNormalizerFromFile(new File("lstm.mod"));

        //Collect training data statistics
        String path = FilenameUtils.concat(DATA_PATH, "physionet2012/");
        String featureBaseDir = FilenameUtils.concat(path, "sequence"); // set feature directory

        CSVSequenceRecordReader recordReader = new CSVSequenceRecordReader(1, ",");
        recordReader.initialize(new FileSplit(new File(featureBaseDir + "/179.csv")));

        DataSetIterator iterator = new SequenceRecordReaderDataSetIterator(recordReader,1,2,86);



        INDArray in =  iterator.next().getFeatures();
        INDArray indArray  =  model.rnnTimeStep(in);
        System.out.println(indArray.toString());
        Gson gson = new Gson();
        System.out.println(gson.toJson(model.predict(in)));
    }

    private static void test4() throws IOException, InterruptedException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("lstm.mod");

        DataNormalization normalizer =  ModelSerializer.restoreNormalizerFromFile(new File("lstm.mod"));          //Note that we are using the exact same normalization process as the training data
        String path = FilenameUtils.concat(DATA_PATH, "physionet2012/");


        String featureBaseDir = FilenameUtils.concat(path, "sequence"); // set feature directory
        String mortalityBaseDir = FilenameUtils.concat(path, "mortality"); // set label directory


        CSVSequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
        testFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 3500, 3999));

        CSVSequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", 3500, 3999));
        testLabels.sequenceRecord();

        SequenceRecordReaderDataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
                1, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        testData.setPreProcessor(normalizer);

        //Evaluate on the test set:
        Evaluation evaluation = model.evaluate(testData);
        log.info(evaluation.stats());
        testData.setPreProcessor(normalizer);
    }

    private static void buildMod() throws IOException, InterruptedException {
        String path = FilenameUtils.concat(DATA_PATH, "physionet2012/");
        DataNormalization normalizer = new NormalizerStandardize();


        String featureBaseDir = FilenameUtils.concat(path, "sequence"); // set feature directory
        String mortalityBaseDir = FilenameUtils.concat(path, "mortality"); // set label directory

        CSVSequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");
        trainFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 0, 3500-1));

        CSVSequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", 0,  3500-1));
        trainLabels.sequenceRecord();
        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
            1, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        CSVSequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
        testFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 3500, 3999));

        CSVSequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", 3500, 3999));

        SequenceRecordReaderDataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
            1, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();
        trainData.setPreProcessor(normalizer);

        testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data


        MultiLayerNetwork net = getModle();
        String str = "Test set at epoch %d: Accuracy = %.2f, F1 = %.2f";
        int nEpochs = 20;

        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);
            //Evaluate on the test set:
            log.info("nEpochs"+i);
            Evaluation evaluation = net.evaluate(testData);
            log.info(evaluation.stats());
            testData.reset();
            trainData.reset();
        }

        boolean saveUpdater = true;

        // ModelSerializer needs modelname, saveUpdater, Location
        ModelSerializer.writeModel(net, new File("lstm.mod"), saveUpdater,normalizer);
    }


    public static MultiLayerNetwork getModle() {
        int NB_INPUTS = 86;



        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)    //Random number generator seed for improved repeatability. Optional.
            .weightInit(WeightInit.XAVIER)
            .updater(new Nadam())
            //.updater(new Nesterovs(0.01,0.9))
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
            .gradientNormalizationThreshold(0.5)
            .list()
            .layer(0, new LSTM.Builder().activation(Activation.SOFTSIGN).nIn(NB_INPUTS).nOut(20).build())
            //.layer(1, new LSTM.Builder().activation(Activation.SOFTSIGN).nIn(30).nOut(18).build())

            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX).nIn(20).nOut(2).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);

        net.init();


        net.setListeners(new ScoreIterationListener(500));   //Print the score (loss function value) every 20 iterations
        return net;

    }



    public static ComputationGraph getModle2() {
        int NB_INPUTS = 86;

        int NB_EPOCHS = 10;
        int RANDOM_SEED = 1234;
        double LEARNING_RATE = 0.005;
        int BATCH_SIZE = 32;
        int LSTM_LAYER_SIZE = 200;
        int NUM_LABEL_CLASSES = 2;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .l2(0.01)
            .graphBuilder()
            .addInputs("in")
            .addLayer("lstm", new GravesLSTM.Builder().nIn(NB_INPUTS).nOut(30).build(), "in")
            .addVertex("lastStep", new LastTimeStepVertex("in"), "lstm")
            .addLayer("out", new OutputLayer.Builder().activation(Activation.SOFTMAX).nIn(30).nOut(2)
                .build(), "lastStep")
            .setOutputs("out")
            .build();


        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        return net;

    }
}
