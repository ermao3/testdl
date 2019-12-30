package demo;

import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by DongBin on 2019/1/25.
 */
public class LSTM2Test {
    static String DATA_PATH = "F:/data";

    private static final Logger log = LoggerFactory.getLogger(LSTM2Test.class);

    static int NB_TRAIN_EXAMPLES = 2000;// number of training examples
    static int NB_TEST_EXAMPLES = 800; // number of testing examples

    public static void main(String[] args) throws Exception {
       // buildMod();



        test3();


        //test2();

        return;
    }


    private static List<Writable> getWritables(List<Double> list) {
        List<Writable> sequence = new ArrayList<>();
        for (Double value : list) {
            sequence.add(new DoubleWritable(value));
        }
        return sequence;
    }

    private static void buildMod() throws IOException, InterruptedException {
        Collection<Collection<Collection<Writable>>> alls = new ArrayList<>();


        for (int i = 0; i < 300; i++) {
            Collection<Collection<Writable>> man = new ArrayList<>();

            List<Double> list = CreateLine.createLine(false);
            int k = 0;
            for (Double value : list) {
                k++;
                List<Writable> sequence = new ArrayList<>();
                sequence.add(new DoubleWritable(value));
                ((ArrayList<Collection<Writable>>) man).add(sequence);
            }
            ((ArrayList<Collection<Collection<Writable>>>) alls).add(man);
        }


        CollectionSequenceRecordReader csrr = new CollectionSequenceRecordReader(alls);
        CollectionSequenceRecordReader csrr_lab = new CollectionSequenceRecordReader(alls);

        SequenceRecordReaderDataSetIterator trainData = new SequenceRecordReaderDataSetIterator(csrr,csrr_lab,
                1, -1, true);

        DataNormalization normalizer = new NormalizerStandardize();

        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();
        trainData.setPreProcessor(normalizer);
        MultiLayerNetwork network = createModel(1, 1);
        int nEpochs = 3;
        for (int i = 0; i < nEpochs; i++) {
            System.out.println("nEpochs" + i);
            network.fit(trainData);
            //Evaluate on the test set:
            trainData.reset();
        }
        boolean saveUpdater = true;
        ModelSerializer.writeModel(network, new File("lstm3.mod"), saveUpdater,normalizer);
    }


    public static void test() throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("lstm2.mod");
        DataNormalization normalizer =  ModelSerializer.restoreNormalizerFromFile(new File("lstm2.mod"));
        Collection<Collection<Collection<Writable>>> alls = new ArrayList<>();


        for (int i = 0; i < 100; i++) {
            Collection<Collection<Writable>> man = new ArrayList<>();

            List<Double> list = CreateLine.createLine(true);
            int k = 0;
            for (Double value : list) {
                k++;
                List<Writable> sequence = new ArrayList<>();
                sequence.add(new DoubleWritable(value));
                ((ArrayList<Collection<Writable>>) man).add(sequence);
            }
            ((ArrayList<Collection<Collection<Writable>>>) alls).add(man);
        }


        CollectionSequenceRecordReader csrr = new CollectionSequenceRecordReader(alls);
        CollectionSequenceRecordReader csrr_lab = new CollectionSequenceRecordReader(alls);

        SequenceRecordReaderDataSetIterator trainData = new SequenceRecordReaderDataSetIterator(csrr,csrr_lab,
                1, -1, true);
        trainData.setPreProcessor(normalizer);
        trainData.reset();


        while (trainData.hasNext()){
            DataSet dataSet = trainData.next();
            double doube = model.score(dataSet);
            //INDArray out =  model.output(dataSet.getFeatures());
            //normalizer.revertFeatures(out);
            //System.out.println(out.toString());

            System.out.println(doube);
        }
    }


    public static void test3() throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("lstm3.mod");
        DataNormalization normalizer =  ModelSerializer.restoreNormalizerFromFile(new File("lstm3.mod"));
        Collection<Collection<Collection<Writable>>> alls = new ArrayList<>();


        for (int i = 0; i < 1; i++) {
            Collection<Collection<Writable>> man = new ArrayList<>();

            List<Double> list = CreateLine.createLine(false);
            int k = 0;
            for (Double value : list) {
                k++;
                List<Writable> sequence = new ArrayList<>();
                sequence.add(new DoubleWritable(value));
                ((ArrayList<Collection<Writable>>) man).add(sequence);
            }
            ((ArrayList<Collection<Collection<Writable>>>) alls).add(man);
        }


        CollectionSequenceRecordReader csrr = new CollectionSequenceRecordReader(alls);
        CollectionSequenceRecordReader csrr_lab = new CollectionSequenceRecordReader(alls);

        SequenceRecordReaderDataSetIterator trainData = new SequenceRecordReaderDataSetIterator(csrr,csrr_lab,
                1, -1, true);
        trainData.setPreProcessor(normalizer);
        trainData.reset();


        while (trainData.hasNext()){

            org.deeplearning4j.nn.layers.recurrent.LSTM lstm = (org.deeplearning4j.nn.layers.recurrent.LSTM) model
                    .getLayer(0);

            INDArray out1 =  lstm.activate(trainData.next().getFeatures(),false,LayerWorkspaceMgr.noWorkspaces());
            System.out.println(out1.toString());

           org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                    = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) model.getLayer(1);

            INDArray out2 =   vae.activate(out1,false,LayerWorkspaceMgr.noWorkspaces());
            //System.out.println(out2.toString());

        }
    }

    public static MultiLayerNetwork createModel(int inputNum, int outputNum) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(123456)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nadam())
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .list()
                .layer(0, new LSTM.Builder().nIn(inputNum).activation(Activation.TANH).nOut(10).build())
                .layer(1, new VariationalAutoencoder.Builder().activation(Activation.LEAKYRELU).
                        encoderLayerSizes(256, 256).decoderLayerSizes(256, 256)
                        .pzxActivationFunction(Activation.IDENTITY)
                        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                        .nIn(10) //Input size:28x28
                        .nOut(10) //Size of the latent variable space: p(z|x) - 32 values
                        .build())
                .layer(2, new RnnOutputLayer.Builder().name("output").nOut(outputNum)
                        .activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

       /* UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);*/
        net.setListeners(new ScoreIterationListener(100));
/*
        net.setListeners(new StatsListener(statsStorage));
*/

        return net;
    }

}
