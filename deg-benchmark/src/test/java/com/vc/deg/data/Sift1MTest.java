package com.vc.deg.data;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class Sift1MTest {

	public static void main(String[] args) throws IOException {
		Path baseDir = Paths.get("c:\\Data\\Feature\\2DGraph\\");
		
		Path baseFile = baseDir.resolve("base.fvecs");
		int seed = 80733299; //(int)(Math.random() * Integer.MAX_VALUE);
		System.out.println("Random seed "+seed);
		Random rnd = new Random(seed);
		float[][] baseData = new float[14][]; 
		for (int i = 0; i < baseData.length; i++) {
			baseData[i] = new float[] { rnd.nextFloat()*13, rnd.nextFloat()*13 };
			System.out.println(Arrays.toString(baseData[i]));
		}
//		float[][] baseData = new float[][] { 
//			{2.83f, 3.3f},
//			{3.48f, 3.58f},
//			{7.47f, 3.95f},
//			{3.05f, 4.92f},
//			{6.97f, 5.39f},
//			{4.09f, 6.31f},
//			{2.47f, 7.18f},
//			{5.07f, 9.69f},
//			{8.14f, 10.05f},
//			{3.91f, 11.01f},
//			{9.13f, 12.19f},
//			{1.36f, 12.42f},
//			{2.51f, 12.6f},
//			{7.07f, 12.91f}
//		};
		Sift1M.fvecs_write(baseData, baseFile);
		
		Path baseTopFile = baseDir.resolve("base_top13.ivecs");
		int[][] baseTopData = new int[baseData.length][];
		for (int i = 0; i < baseTopData.length; i++) {
			int baseIndex = i;
			baseTopData[i] = IntStream.range(0, baseData.length).filter(idx -> idx != baseIndex).boxed().sorted((idx1, idx2) -> Double.compare(distance(baseData[baseIndex], baseData[idx1]), distance(baseData[baseIndex], baseData[idx2]))).mapToInt(Integer::intValue).toArray();
		}
		Sift1M.ivecs_write(baseTopData, baseTopFile);
		
		Path exploreFile = baseDir.resolve("explore.fvecs");
		Sift1M.fvecs_write(baseData, exploreFile);		
		
		Path exploreEntryFile = baseDir.resolve("explore_entry_node.ivecs");
		int[][] exploreEntryNodeIdx = new int[baseData.length][];
		for (int i = 0; i < exploreEntryNodeIdx.length; i++) 
			exploreEntryNodeIdx[i] = new int[] {i};
		Sift1M.ivecs_write(exploreEntryNodeIdx, exploreEntryFile);
				
		Path exploreGtFile = baseDir.resolve("explore_gt.ivecs");
		int[][] exploreGtData = new int[baseData.length][];
		for (int i = 0; i < exploreGtData.length; i++) 
			exploreGtData[i] = Arrays.copyOf(baseTopData[i], 10);
		Sift1M.ivecs_write(exploreGtData, exploreGtFile);
		
		Path queryFile = baseDir.resolve("query.fvecs");
		float[][] queryData = new float[][] { 
			{10.23f, 8.33f} 
		};
		Sift1M.fvecs_write(queryData, queryFile);
		
		Path queryGtFile = baseDir.resolve("query_gt.ivecs");
		int[][] queryGtData = new int[][] { 
			IntStream.range(0, baseData.length).boxed().sorted((idx1, idx2) -> Double.compare(distance(queryData[0], baseData[idx1]), distance(queryData[0], baseData[idx2]))).mapToInt(Integer::intValue).limit(5).toArray() 
		};
		Sift1M.ivecs_write(queryGtData, queryGtFile);
	}
	
	/**
	 * Compute euclidean distance
	 * 
	 * @param vec1
	 * @param vec2
	 * @return
	 */
	public static double distance(float[] vec1, float[] vec2) {
		double result = 0;
		for (int i = 0; i < vec1.length; i++) {
			double diff = vec1[i] - vec2[i];
			result += diff*diff;
		}
		return Math.sqrt(result);
	}
}
