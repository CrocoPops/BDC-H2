import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;
import scala.Tuple3;
import java.io.*;
import java.util.*;


public class G008HW2 {

    private static JavaSparkContext sc;

    public static void main(String[] args) throws IOException {
        // Check if the number of arguments is correct
        if (args.length != 4)
            throw new IllegalArgumentException("Wrong number of params!");

        // Variables
        int M = Integer.parseInt(args[1]);
        int K = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);

        // Time measurement
        long startTime;
        long endTime;
        long totalTime;

        // Printing arguments
        System.out.println(args[0] + " M=" + args[1] + " K=" + args[2] + " L=" + args[3] + " ");

        // Creating the Spark context
        SparkConf conf = new SparkConf(true).setAppName("OutlierDetector");
        sc = new JavaSparkContext(conf);

        // Divide the inputFile in L partitions
        JavaRDD<String> rawData = sc.textFile(args[0]).repartition(L).cache();
        JavaPairRDD<Float, Float> inputPoints = rawData.mapToPair(document -> {
            // Parse the points' coordinates
            String[] cord = document.split(",");
            Tuple2<Float, Float> point = new Tuple2<>(Float.parseFloat(cord[0]), Float.parseFloat(cord[1]));

            return new Tuple2<>(point._1(), point._2());
        });
        // Print the number of points
        System.out.println("Number of points = " + inputPoints.count());

        // Compute the radius of the K-center clustering, the maximum distance of a point from its closest center
        float R = MRFFT(inputPoints, K);
        startTime = System.currentTimeMillis();
        // Compute the approximate outliers
        MRApproxOutliers(inputPoints, R, M);
        endTime = System.currentTimeMillis();
        totalTime = endTime - startTime;
        System.out.println("Running time of MRApproxOutliers  = " + totalTime + " ms");

    }

    public static double distanceTo(Tuple2<Float, Float> p1, Tuple2<Float, Float> p2) {
        float deltaX = p1._1() - p2._1();
        float deltaY = p1._2() - p2._2();
        return Math.pow(deltaX, 2) + Math.pow(deltaY, 2);
    }

    /**
     * Sequential Farthest-First Traversal algorithm
     * @param listOfPoints - set of points
     * @param K - number of clusters
     * @complex O(|P|*K)
     * @return C - a set of K centers of the cluster
     */
    public static List<Tuple2<Float, Float>> SequentialFFT(List<Tuple2<Float, Float>> listOfPoints, int K) {
        List<Tuple2<Float, Float>> C = new ArrayList<>();
        Random random = new Random();
        // Choosing randomly the first center
        Tuple2<Float, Float> p = listOfPoints.get(random.nextInt(listOfPoints.size()));
        C.add(p);
        // O(|P|)
        Map<Tuple2<Float, Float>, Double> distances = new HashMap<>();
        // Compute for every point the distances from the center
        for(Tuple2<Float, Float> point: listOfPoints) {
            distances.put(point, distanceTo(point, p));
        }

        //O(|P|*K)
        for(int i = 1; i < K; i++) {
            Tuple2<Float, Float> cand = null;
            double maxDistance = Double.MIN_VALUE;
            // Farthest point becomes a center
            for(Map.Entry<Tuple2<Float, Float>, Double> entry: distances.entrySet()) {
                if(entry.getValue() > maxDistance) {
                    maxDistance = entry.getValue();
                    cand = entry.getKey();
                }
            }
            C.add(cand);
            distances.remove(cand);

            // The distances must be recomputed
            // Some points now could be closer to the new center then to the previous one
            for(Map.Entry<Tuple2<Float, Float>, Double> entry: distances.entrySet()) {
                assert cand != null;
                double distance = distanceTo(entry.getKey(), cand);
                if(distance < entry.getValue()) {
                    distances.put(entry.getKey(), distance);
                }
            }
        }
        return C;
    }

    /**
     * MapReduce FFT-Algorithm
     * @param inputPoints - set of input points stored in an RDD
     * @param K - number of clusters
     * @return R - radius R
     */
    public static float MRFFT(JavaPairRDD<Float, Float> inputPoints, int K) {
        long startTime;
        long endTime;
        // Round 1
        // Map - For each partition P map all the points (Tuple2<Float, Float>) in a unique List<Tuple2<Float, Float>>
        // Reduce - for every partition run SequentialFFT on Pi to determine a set Ti of K centers
        startTime = System.currentTimeMillis();
        JavaRDD<Tuple2<Float, Float>> centersPartition = inputPoints.mapPartitions(pointsIterator -> {
            List<Tuple2<Float, Float>> pointList = new ArrayList<>();
            while(pointsIterator.hasNext()){
                pointList.add(pointsIterator.next());
            }

            return SequentialFFT(pointList, K).iterator();
        }).cache(); // Save in local memory

        // Force Spark to run the Round 1 doing an operation
        long xTemp = centersPartition.count();

        endTime = System.currentTimeMillis();
        System.out.println("Running time of MRFFT Round 1 = " + (endTime - startTime) + " ms");
        // Round 2
        // Map - empty
        // Reduce - gather the coreset T of size L*K and run, using a single reducer, SequentialFFT on T to determine a set S of centers
        // of K centers and return S as output
        startTime = System.currentTimeMillis();
        // Collect all the centers of each Ti on T
        List<Tuple2<Float, Float>> T = centersPartition.collect();
        // Compute the SequentialFFT() on the entire RDD
        List<Tuple2<Float, Float>> centers = SequentialFFT(T, K);
        endTime = System.currentTimeMillis();
        System.out.println("Running time of MRFFT Round 2 = " + (endTime - startTime) + " ms");


        // Round 3
        // Compute and returns the radius R of the clustering induced by the centers that is dist(x, C) for every x in P.
        startTime = System.currentTimeMillis();
        Broadcast<List<Tuple2<Float, Float>>> centersBroadcast = sc.broadcast(centers);
        float R = inputPoints.mapToDouble(point -> {
            List<Tuple2<Float, Float>> centersList = centersBroadcast.getValue();
            double minDistance = Double.MAX_VALUE;

            for(Tuple2<Float, Float> center: centersList) {
                double distance = distanceTo(center, point);
                if(distance < minDistance)
                    minDistance = distance;
            }
            return minDistance;
        }).reduce(Math::max).floatValue();
        R = (float)Math.sqrt(R);
        endTime = System.currentTimeMillis();
        System.out.println("Running time of MRFFT Round 3 = " + (endTime - startTime) + " ms");
        System.out.println("Radius = " + R);
        return R;
    }
    public static void MRApproxOutliers(JavaPairRDD<Float, Float> inputPoints, float D, int M) {
        // ROUND 1
        // Mapping each pair (X,Y) into ((X,Y), 1)
        JavaPairRDD<Tuple2<Integer, Integer>, Long> cell = inputPoints.flatMapToPair(point -> { // <-- MAP PHASE (R1)

            // Compute the cells coordinates
            double lambda = D / (2 * Math.sqrt(2));

            // Finding cell coordinates
            Tuple2<Integer, Integer> cellCoordinates = new Tuple2<>(
                    (int) Math.floor(point._1() / lambda),
                    (int) Math.floor(point._2() / lambda)
            );

            return Collections.singletonList(new Tuple2<>(cellCoordinates, 1L)).iterator();
        }).reduceByKey(Long::sum).cache();

        // Saving the non-empty cells in a local structure
        Map<Tuple2<Integer, Integer>, Long> nonEmptyCells = cell.collectAsMap();

        // Round 2
        // Adding information on N3 and N7 for each K-V pair
        JavaPairRDD<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> cellNeighbors = cell.mapToPair(pair -> {
            int i = pair._1()._1();
            int j = pair._1()._2();
            long totalCount = pair._2();
            long N3 = 0L;
            long N7 = 0L;

            for (int dx = -3; dx <= 3; dx++) {
                for (int dy = -3; dy <= 3; dy++) {
                    Tuple2<Integer, Integer> neighborKey = new Tuple2<>(i + dx, j + dy);
                    Long neighborCount = nonEmptyCells.get(neighborKey);

                    if (neighborCount != null) {
                        if ((Math.abs(dx) <= 1) && (Math.abs(dy) <= 1))
                            N3 += neighborCount;
                        N7 += neighborCount;
                    }
                }
            }

            Tuple3<Long, Long, Long> counts = new Tuple3<>(totalCount, N3, N7);
            return new Tuple2<>(new Tuple2<>(i, j), counts);
        }).cache();

        // Number of sure (D, M) - outliers
        long sureOutliers = 0;
        for (Tuple2<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> i : cellNeighbors.filter(triple -> triple._2()._3() <= M).collect())
            sureOutliers += i._2()._1();
        System.out.println("Number of sure outliers = " + sureOutliers);

        // Number of uncertain points
        long uncertainOutliers = 0;
        for (Tuple2<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> i : cellNeighbors.filter(triple -> triple._2()._2() <= M && triple._2()._3() > M).collect())
            uncertainOutliers += i._2()._1();
        System.out.println("Number of uncertain points = " + uncertainOutliers);
    }
}