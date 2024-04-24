import org.apache.spark.api.java.JavaPairRDD;
import scala.Tuple2;
import scala.Tuple3;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;


public class G008HW2 {

    public static void main(String[] args) throws IOException {
        System.out.println("ciao");
    }

    public static double distance(List<Double> point1, List<Double> point2) {
        // Calculate Euclidean distance between two points
        double sum = 0;
        for (int i = 0; i < point1.size(); i++) {
            sum += Math.pow(point1.get(i) - point2.get(i), 2);
        }
        return Math.sqrt(sum);
    }

    public static List<List<Double>> SequentialFFT(List<List<Double>> points, int K) {

        // Choose the first point arbitrarily
        List<List<Double>> centers = new ArrayList<>();
        centers.add(points.get(0));

        // IDEA: maximize the minimum distance between a point and the closest center

        // While the number of centers is less than K
        while (centers.size() < K) {

            // Find the point farthest from the current centers
            List<Double> farthestPoint = null;
            double maxDistance = Double.MIN_VALUE;

            for (List<Double> point : points) {
                double minDistance = Double.MAX_VALUE;

                // Find the closest center distance
                for (List<Double> center : centers)
                    minDistance = Math.min(minDistance, distance(point, center));

                // With minDistance we have that the point
                if (minDistance > maxDistance) {
                    maxDistance = minDistance;
                    farthestPoint = point;
                }
            }

            // Add the farthest point to the centers list
            centers.add(farthestPoint);
        }

        return centers;
    }


    public static void MRApproxOutliers(JavaPairRDD<Float, Float> inputPoints, float D, int M) {
        // ROUND 1
        // Mapping each pair (X,Y) into ((X,Y), 1)
        JavaPairRDD<Tuple2<Integer, Integer>, Long> cell = inputPoints.flatMapToPair(point -> { // <-- MAP PHASE (R1)

            // Compute the cells coordinates
            double lambda = D/(2*Math.sqrt(2));

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

            for(int dx = -3; dx <= 3; dx++) {
                for(int dy = -3; dy <= 3; dy++) {
                    Tuple2<Integer, Integer> neighborKey = new Tuple2<>(i + dx, j + dy);
                    Long neighborCount = nonEmptyCells.get(neighborKey);

                    if(neighborCount != null){
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
        for(Tuple2<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> i : cellNeighbors.filter(triple -> triple._2()._3() <= M).collect())
            sureOutliers += i._2()._1();
        System.out.println("Number of sure outliers = " + sureOutliers);

        // Number of uncertain points
        long uncertainOutliers = 0;
        for(Tuple2<Tuple2<Integer, Integer>, Tuple3<Long, Long, Long>> i : cellNeighbors.filter(triple -> triple._2()._2() <= M && triple._2()._3() > M).collect())
            uncertainOutliers += i._2()._1();
        System.out.println("Number of uncertain points = " + uncertainOutliers);

    }

}
