package com.vc.deg.viz.om;

import java.util.Arrays;

import it.unimi.dsi.fastutil.floats.FloatArrayList;
import it.unimi.dsi.fastutil.floats.FloatList;

public class DistancePreservation {

	protected static final double[] POWERS = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
	
	private static boolean SORT_Equal_2D_DISTS = true; // false: average equal 2D dists
	private float L2_p = 1;    
	
	
	private double[] D_HD;
	private double[] D_2D;
	private double D_mean2;
	
	private double[] NDP_power;
	private final int power_index;
	
	private final int numNeighbors;
	
	/**
	 * 
	 * @param featureMap
	 * @param mapWidth
	 * @param mapHeight
	 * @param doWrap
	 */
	public DistancePreservation(float[][] features, double power) {
		
		this.power_index = Arrays.binarySearch(POWERS, power);
		if(power_index < 0)
			throw new RuntimeException("Power "+power+" not found in POWERS Array " + Arrays.toString(POWERS));
		
		// count the number of feature vectors
		numNeighbors = features.length; // (int) Arrays.stream(features).filter(Objects::nonNull).count(); // needs support for holes
				
		D_HD = calc_dmean_and_Dh(features);		
		D_mean2 = D_HD[numNeighbors-1]; // global mean 
	}
	
	private double[] calc_dmean_and_Dh(float[][] featureMap) {
		
		int numPics = featureMap.length;
		
		float[][] sortedDistsHD = new float[numPics][numPics];
		
		for (int i = 0; i < numPics; i++) {
			final float[] ele1 = featureMap[i];
			final float[] fv1 = ele1;	

			for (int j = i+1; j < numPics; j++) {
		        final float[] ele2 = featureMap[j];
		        final float[] fv2 = ele2;
				
				float dist;
				if (L2_p == 2)
					dist = getSSEDistance(fv1, fv2);
				else if (L2_p == 1)
					dist = getL2Distance(fv1, fv2);
				else 
					dist = getL2pDistance(fv1, fv2, L2_p);
							
				sortedDistsHD[i][j] = sortedDistsHD[j][i] = dist; 
			}		
		}
		
		for (int i=0; i < numPics; i++)	
			Arrays.sort(sortedDistsHD[i]);
		
		final double[] D_HD = new double[numNeighbors];
		
		for (int n = 0; n < numNeighbors; n++) 
			for (int i = 0; i < numPics; i++) 
				D_HD[n] += sortedDistsHD[i][n];
		
		for (int n = 1; n < numNeighbors; n++) 
			D_HD[n] += D_HD[n-1];		
		
		for (int n = 1; n < numNeighbors; n++) 
			D_HD[n] /= (long)n*numPics;		
		
		return D_HD; 
	}


	
	private double[] calc_Dist2D(float[][] featureMap, int mapWidth, int mapHeight, boolean doWrap) {
		final int maxDist2D = (mapHeight-1)*(mapHeight-1) + (mapWidth-1)*(mapWidth-1) + 1;
		
		D_2D = new double[numNeighbors];
		final float[][] sortedDists2D = new float[mapWidth*mapHeight][numNeighbors];
		
		
		//IntStream.range(0, mapWidth).parallel().forEach((int x1) -> {
		for (int x1 = 0; x1 < mapWidth; x1++) {
			for (int y1 = 0; y1 < mapHeight; y1++) {  // Schleife über alle Positionen
				
				final int pos1 = y1*mapWidth + x1;
				if (featureMap[pos1] == null)
					continue;
				
				final float[] ele1 = featureMap[pos1];
				final float[] fv1 = ele1;	

				final FloatList[] al = new FloatArrayList[maxDist2D];
				  
		        for (int i = 0; i < maxDist2D; i++) 
		        	al[i] = new FloatArrayList();
				
				for (int x2 = 0; x2 < mapWidth; x2++) {
					int dx = Math.abs(x1-x2);
					if (doWrap) 
						dx = Math.min(dx, mapWidth-dx);

					for (int y2 = 0; y2 < mapHeight; y2++) {
						final int pos2 = y2*mapWidth + x2;
						if (featureMap[pos2] == null)
							continue;

						final float[] ele2 = featureMap[pos2];
						
						if(ele1 == ele2)  // skip self reference
							continue;

						float[] fv2 = ele2;
						int dy = Math.abs(y1-y2);
						if (doWrap) 
							dy = Math.min(dy, mapHeight-dy);

						final int dist2D = dx*dx + dy*dy;
						float dist;
						if (L2_p == 2)
							dist = getSSEDistance(fv1, fv2);
						else if (L2_p == 1)
							dist = getL2Distance(fv1, fv2);
						else 
							dist = getL2pDistance(fv1, fv2, L2_p);
												
						al[dist2D].add(dist);
					}
				}
				
				int k = 1;
				for (int i = 0; i < al.length; i++) {
					if (al[i].size() > 0) {
						if (SORT_Equal_2D_DISTS) {
							al[i].sort( Float::compare );
							for (int j = 0; j < al[i].size(); j++) 
								sortedDists2D[pos1][k++] = al[i].getFloat(j);
						}
						else {
							float avg = 0;
							for (int j = 0; j < al[i].size(); j++)
								avg += al[i].getFloat(j);
							avg /=  al[i].size();
							for (int j = 0; j < al[i].size(); j++) 
								sortedDists2D[pos1][k++] = avg;
						}
					}
		        }	
			}
		} // );
		
		for (int pos = 0; pos < sortedDists2D.length; pos++) 
			for (int k = 0; k < numNeighbors; k++) 
				D_2D[k] += sortedDists2D[pos][k];
		
		for (int j = 1; j < numNeighbors; j++) 
			D_2D[j] += D_2D[j-1];		
		
		for (int j = 1; j < numNeighbors; j++) 
			D_2D[j] /= (long)j*numNeighbors;	
	
		return D_2D;
	}
	
	public static float getSSEDistance(final float[] actFeatureData, final float[] searchFeatureData) {
		float dist = 0;
		for (int i = 0; i < actFeatureData.length; i++) {
			float d = actFeatureData[i] - searchFeatureData[i];
			dist += d*d;
		}
		return dist;
	}
	
	public static float getL2Distance(final float[] actFeatureData, final float[] searchFeatureData) {
		float dist = 0;
		for (int i = 0; i < actFeatureData.length; i++) {
			float d = actFeatureData[i] - searchFeatureData[i];
			dist += d*d;
		}
		return (float) Math.sqrt(dist);
	}
	
	public static float getL2pDistance(final float[] actFeatureData, final float[] searchFeatureData, final float L2_p) {
		float dist = 0;
		for (int i = 0; i < actFeatureData.length; i++) {
			float d = actFeatureData[i] - searchFeatureData[i];
			dist += d*d;
		}
		return (float) Math.pow(Math.sqrt(dist), L2_p);
	}
	
	private double[] computeQuality(double[] D_2D, double[] powers) {

		double[] DP_Sort = new double[powers.length];
		double[] DP_Opt = new double[powers.length];

		if (D_mean2 == 0) { // all vectors are the same
			double[] result = new double[powers.length];
			for (int p = 0; p < powers.length; p++)
				result[p] = 1;
			return result; 
		}		
		
		// compare HD and 2D qualities
		for (int k = 1; k < numNeighbors; k++) {
			
			double DP_k = Math.max(0, 1 - D_2D[k]/D_mean2);
			double DP_k_opt =  1 - D_HD[k]/D_mean2;
			
			//System.out.println(DP_k);
			//System.out.println(DP_k_opt);
			//System.out.println();			
			
			for (int p = 0; p < powers.length; p++) {
				double power = powers[p];
				DP_Sort[p] += Math.pow(DP_k, power); 
				DP_Opt[p]  += Math.pow(DP_k_opt, power); 				
			}
		}

		double[] result = new double[powers.length];
		for (int p = 0; p < powers.length; p++) {
			double power = powers[p];
			double DP_Sort_val = Math.pow(DP_Sort[p], 1./power);
			double DP_Opt_val = Math.pow(DP_Opt[p], 1./power);
			result[p] = DP_Sort_val / DP_Opt_val;
		}

		return result;  // NDP(p)
	}

	public double computeQuality(float[][] imageGrid, int gridColumns, int gridRows, boolean doWrap) {		
		final double[] D_2D = calc_Dist2D(imageGrid, gridColumns, gridRows, doWrap);		
		NDP_power = computeQuality(D_2D, POWERS);		
		return NDP_power[power_index];
	}
	
	
	public double[] getComputeQuality() {		
		if(NDP_power == null) 
			throw new RuntimeException("Please call computeQuality first.");		
		return NDP_power;
	}
	
	@Override
	public String toString() {
		double pow = POWERS[power_index]; 
		String sPower = String.format("%.0f", pow);
		if (SORT_Equal_2D_DISTS)
			return "DPQ_"+sPower;
		else
			return "DPQ\u2CBA_"+sPower;
	}	
}
