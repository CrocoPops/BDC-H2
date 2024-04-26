import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.jetbrains.annotations.NotNull;
import scala.Tuple2;
import scala.Tuple3;
import java.io.*;
import java.util.*;


public class G008HW2 {

    private static List<Point> centers;

    public void main(String[] args) throws IOException {
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

        // Printing CLI arguments
        System.out.println(args[0] + " M=" + args[1] + " K=" + args[2] + " L=" + args[3] + " ");

        // Read all points in the file and add them to the list
        Scanner scanner = new Scanner(new File(args[0]));
        List<Point> listOfPoints = new ArrayList<>();
        while (scanner.hasNextLine()) {
            String[] cords = scanner.nextLine().split(",");
            listOfPoints.add(new Point(Float.parseFloat(cords[0]), Float.parseFloat(cords[1])));
        }

        // Print the number of points
        System.out.println("Number of points = " + listOfPoints.size());

        // Creating the Spark context and calling outliers approximate computation
        SparkConf conf = new SparkConf(true).setAppName("OutlierDetector");
        try (JavaSparkContext sc = new JavaSparkContext(conf)) {
            sc.setLogLevel("ERROR");
            // Divide the inputFile in L partitions (each line is assigned to a specific partition
            JavaRDD<String> rawData = sc.textFile(args[0]).repartition(L).cache();
            JavaPairRDD<Float, Float> inputPoints = rawData.mapToPair(document -> {
                String[] cord = document.split(",");
                Tuple2<Float, Float> point = new Tuple2<>(Float.parseFloat(cord[0]), Float.parseFloat(cord[1]));

                return new Tuple2<>(point._1(), point._2());
            });
            startTime = System.currentTimeMillis();
            float R = MRFFT(inputPoints, K);
            endTime = System.currentTimeMillis();
            startTime = System.currentTimeMillis();
            // D is the radius of the K-center clustering, the maximum distance of a point from its closest center
            MRApproxOutliers(inputPoints, R, M, K);
            endTime = System.currentTimeMillis();
            totalTime = endTime - startTime;
            System.out.println("Running time of MRApproxOutliers  = " + totalTime + "ms");
        }
    }

    /**
     * Sequential Farthest-First Traversal algorithm
     * @param listOfPoints - set of points
     * @param K - number of clusters
     * @complex O(|P|*K)
     * @return C - a set of K centers of the cluster
     */
    public static List<Point> SequentialFFT(List<Point> listOfPoints, int K) {
        List<Point> C = new ArrayList<>();
        Random random = new Random();
        // Choosing randomly the first center
        Point p = listOfPoints.get(random.nextInt(listOfPoints.size()));
        C.add(p);
        listOfPoints.remove(p);
        // O(|P|)
        Map<Point, Double> distances = new HashMap<>();
        // Compute for every point the distances from its closest center
        for(Point point: listOfPoints) {
            double maxDistance = Double.MIN_VALUE;
            // O(1)
            for(Point center: C) {
                double distance = point.distanceTo(center);
                if(distance > maxDistance) {
                    maxDistance = distance;
                }
            }
            distances.put(point, maxDistance);
        }
        //O(|P|*K)
        for(int i = 1; i < K; i++) {
            Point cand = null;
            double maxDistance = Double.MIN_VALUE;
            // Farthest point becomes a center
            for(Map.Entry<Point, Double> entry: distances.entrySet()) {
                if(entry.getValue() > maxDistance) {
                    maxDistance = entry.getValue();
                    cand = entry.getKey();
                }
            }
            C.add(cand);
            listOfPoints.remove(cand);
            distances.remove(cand);

            // The distances must be recomputed
            // Some points now could be closer to the new center then to the previous one
            for(Map.Entry<Point, Double> entry: distances.entrySet()) {
                assert cand != null;
                double distance = entry.getKey().distanceTo(cand);
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
        // Map - Partition P arbitrarily into l subsets of equal size
        // Reduce - for every partition run FFT on Pi to determine a set Ti of K centers
        int l = K; // TODO: change here, l should be L taken in input, all the partition step must be moved into the main
        startTime = System.currentTimeMillis();
        JavaPairRDD<Integer, List<Point>> pointPartitions = inputPoints.mapToPair(point -> {
            Random random = new Random();
            return new Tuple2<>(random.nextInt(l), point);
        }).groupByKey().mapValues(iterable -> {
            List<Point> pointsPartition = new ArrayList<>();
            iterable.forEach(p -> pointsPartition.add(new Point(p._1(), p._2())));
            return SequentialFFT(pointsPartition, K);
        });
        endTime = System.currentTimeMillis();
        System.out.println("Round 1 - " + (endTime - startTime) + " ms.");
        // Round 2
        // Map - empty
        // Reduce - gather the coreset T of size l*k and run, using a single reducer, FFT on T to determine a set S
        // of K centers and return S as output
        startTime = System.currentTimeMillis();
        JavaRDD<Point> T = pointPartitions.values().flatMap(List::iterator);
        centers = SequentialFFT(T.collect(), K);
        endTime = System.currentTimeMillis();
        System.out.println("Round 2 - " + (endTime - startTime) + " ms.");
        // Round 3
        // Compute and returns the radius R of the clustering induced by the centers that is dist(x, C) for every x in P.
        startTime = System.currentTimeMillis();
        Broadcast<List<Point>> centersBroadcast = new JavaSparkContext(inputPoints.context()).broadcast(centers);
        float R = inputPoints.mapToDouble(point -> {
            List<Point> centersList = centersBroadcast.getValue();
            double minDistance = Double.MAX_VALUE;

            for(Point center: centersList) {
                double distance = center.distanceTo(new Point(point._1(), point._2()));
                if(distance < minDistance)
                    minDistance = distance;
            }
            return minDistance;
        }).reduce(Math::max).floatValue();
        endTime = System.currentTimeMillis();
        System.out.println("Round 3 - " + (endTime - startTime) + " ms.");

        return R;
    }
    public static void MRApproxOutliers(JavaPairRDD<Float, Float> inputPoints, float R, float D, int M) {
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

        // First K cells in non-decreasing order of cell size
        // IDEA: K is computed looking at the number of centers
        List<Tuple2<Long, Tuple2<Tuple2<Integer, Integer>, Long>>> topKCells = cell.mapToPair(
                tuple -> new Tuple2<>(tuple._2(), tuple)
        ).sortByKey(true).take((int) R);

        for (Tuple2<Long, Tuple2<Tuple2<Integer, Integer>, Long>> i_cell : topKCells)
            System.out.println("Cell: " + i_cell._2()._1() + " Size = " + i_cell._1());
    }
}

// Class used as struct to contain information about the points
class Point implements Comparable<Point>{
    float x;
    float y;
    Long nearby;

    public Point(float x, float y){
        this.x = x;
        this.y = y;
        this.nearby = 1L;
    }

    public double distanceTo(Point other) {
        float deltaX = other.x - this.x;
        float deltaY = other.y - this.y;
        return Math.pow(deltaX, 2) + Math.pow(deltaY, 2);
    }

    @Override
    public int compareTo(@NotNull Point o) {
        return Long.compare(this.nearby, o.nearby);
    }

}


