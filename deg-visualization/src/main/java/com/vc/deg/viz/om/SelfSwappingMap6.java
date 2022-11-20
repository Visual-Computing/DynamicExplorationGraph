/* (c) Copyright 2017 by K. Barthel, N. Hezel
 * All rights reserved
 */

package com.vc.deg.viz.om;

import java.util.Random;

import com.koloboke.collect.set.IntSet;
import com.koloboke.collect.set.hash.HashIntSets;

/**
 * @author barthel and hezel
 */
public class SelfSwappingMap6 {
	
	public static final String version = "SSM6";	// corresponds to Kai's sorterName in UniversalSorter

	// SSM settings  // optimale Werte hängen von der Komplexität der L2-Distanzberechnung ab
	private int numIterations = 6;      //  5  // maximum iterations for SSM swaps 
	private int numSwapPositions = 22;  // 22  // number of swap positions 
	private int stride = 4;             //  4  // step width for SSM swap positions
	private boolean doWrap = true;             // indicates if the SOM should be wrapped or not
	
	private boolean doSimultaneousSwap = false;
	
	// data 
	private MapPlace[] mapPlaces;	
	
	// SOM
	private int nX, nY, dims;
	private float[][] som;		
	private float[] weight;					// weights for the filtering process
	private final float weightHole = 1f;    // evtl. an Lückenanteil angepassen
	private final float weightSwappable = 10f;  
	private final float weightNonSwappable = 1000f;  
	
	
	// integral image 
	private int nXI, nYI;       // size of the integral image
	private float[][] somI;		// Integral image of the visual SOM
	private float[] weightI;	// Integral image of the weights for the filtering
	
	private int[][] distLut  = new int[numSwapPositions][numSwapPositions];
	private int oldRadius;
	private static Random random;
	
	
	public SelfSwappingMap6(MapPlace[] mapPlaces, int mapWidth, int mapHeight, int seed) { 
		
		this.mapPlaces = mapPlaces;
		nX = mapWidth;	// size of the SOM
		nY = mapHeight;
		
		int numFVs = 0;         		
		for (int i = 0; i < mapPlaces.length; i++) {  // count the number of fvs to swap
			if (mapPlaces[i] != null && mapPlaces[i].isSwapable()) 
				numFVs++;
		}
		
	    if (numFVs < numSwapPositions) { 
	        numSwapPositions = numFVs; 
	        distLut  = new int[numSwapPositions][numSwapPositions]; 
	      } 
		
		for (int i = 0; i < mapPlaces.length; i++) {  // get dimension of fv
			if (mapPlaces[i] != null) {
				dims = mapPlaces[i].getFloatFeature().length;	
				break;
			}
		}

		som = new float[nX*nY][dims];
		weight = new float[nX*nY];	

		oldRadius = -1;	
		random = new Random(seed);	
	}
	
	public void setNumIterations(int numIterations) {
		this.numIterations = numIterations;
	}
	
	public void setStride(int stride) {
		this.stride = stride;
	}
	
	public void setNumSwapPositions(int numSwapPositions) {
		if (numSwapPositions <= 0 ) {
			numSwapPositions = 0;
			for (int i = 0; i < mapPlaces.length; i++) {  // count the number of fvs to swap
				if (mapPlaces[i] == null || mapPlaces[i].isSwapable()) 
					numSwapPositions++;
			}
			doSimultaneousSwap = true;
		}
		this.numSwapPositions = numSwapPositions;
		distLut  = new int[numSwapPositions][numSwapPositions];
	}
	
	public void setWrap(boolean wrap) {
		this.doWrap = wrap;
	}
	
	
	/**
	 * Run the SSM algorithm
	 */
	public void run() {
		copyFeatureVectorsToSom();

		int bs = Math.max(nX, nY); 
		filterSom(bs);
		
		do {	
			bs /= 2; 
			float step = bs/2f/numIterations;
			
			for (int iter = 0; iter < numIterations; iter++) {
				int radius = Math.max(1, (int)(bs-iter*step));
				
				if (doSimultaneousSwap) 
					swapAll();
				else	
					checkStridedSwaps(radius); 
				
				copyFeatureVectorsToSom();
				filterSom(radius); 
			}
		}
		while (bs > 1); 
	}
	
	

	// -----------------------------------------------------------------------------------------------------------
	// -----------------------------------------------------------------------------------------------------------
	// -----------------------------------------------------------------------------------------------------------
	
	
	/** Arranges the given amount of images according to their visual similarity
	 * on a virtual 2D canvas. The position of each image on the resulting
	 * canvas is saved in the corresponding PicWrapper instance.
	 * <br><br>
	 * The arrangement is an iterative process, so several calls of this method
	 * are necessary to get proper results. 
	 * <br><br>
	 * Internal: <br>
	 * 	- write {@link #weight} <br>
	 *  - read {@link #som}, {@link #map}, {@link #inUse}, {@link #means}, {@link #perm}
	 **/
	
	private void checkStridedSwaps(int radius) {  
		
		for (int y = 0; y < nY; y+=stride) {  
			for (int x = 0; x < nX; x+=stride) {
				int pos = y*nX+x;
				if (mapPlaces[pos] != null && mapPlaces[pos].isSwapable())  {
					int[] swapPositions = getSwapPositions(2*radius, y, x);
					checkMultipleSwaps(swapPositions);	
				}
			}
		}
	}
	
	private void copyFeatureVectorsToSom() {
		for (int pos = 0; pos < nX*nY; pos++)  {
			
			MapPlace mapPlace =  mapPlaces[pos];
			if(mapPlace != null) {         
				float[] fv = mapPlace.getFloatFeature();
				float[] actSom = som[pos];
				
				// höheres Gewicht für fixierte Bilder
				float w = (mapPlace.isSwapable()) ? weightSwappable : weightNonSwappable; 
				for (int d = 0; d < dims; d++)
					actSom[d] = w*fv[d];
				weight[pos] = w;  
			}
			else { // Geringes Gewicht für Lücken 
				float[] actSom = som[pos];
				for (int d = 0; d < dims; d++)
					actSom[d] = weightHole*actSom[d];
				weight[pos] = weightHole;  
			}
		}
	}
	
	
	private int[] getSwapPositions(int searchRad, int y0, int x0) {
		x0 += 20*nX - searchRad;   // + 20*nX wegen wrap bzw. modulo 
		y0 += 20*nY - searchRad;   // 20 falls Aspectratio sehr ungleich ist

		IntSet swapPositions = HashIntSets.newUpdatableSet(numSwapPositions);
		int tries = 2 * numSwapPositions;
		do  {
			tries--;
			int dx = random.nextInt(2*searchRad+1);
			int dy = random.nextInt(2*searchRad+1); 
			int x = (x0 + dx) % nX;
			int y = (y0 + dy) % nY;
			int pos = y*nX + x; 
			if (mapPlaces[pos] == null || mapPlaces[pos].isSwapable()) 
				swapPositions.add(pos);
		}
		while (swapPositions.size() < numSwapPositions && tries > 0);
		
		if (swapPositions.size() == numSwapPositions)
			return swapPositions.toIntArray();

		// falls nicht genügend Positionen gefunden wurden, die swappable sind, 
		// dann weiter entfernt nach weiteren Positionen suchen
		int searchFactor = 4;
		tries = 2 * numSwapPositions;
		x0 += 2*nX - searchRad; 
		y0 += 2*nY - searchRad;

		while (swapPositions.size() < numSwapPositions) {		
			tries--;	
			int dx = random.nextInt(searchFactor*searchRad+1); 
			int dy = random.nextInt(searchFactor*searchRad+1); 
			int x = (x0 + dx) % nX;
			int y = (y0 + dy) % nY;
			int pos = y*nX + x; 
			if (mapPlaces[pos] == null || mapPlaces[pos].isSwapable()) 
				swapPositions.add(pos);
			if (tries == 0) {
				searchFactor += 2;
				x0 += 2*nX - searchRad; 
				y0 += 2*nY - searchRad;
				tries = 2 * numSwapPositions;
			}
		}		
		return swapPositions.toIntArray();
	}

	
	/** Finds the best swap (permutation) of num images taking into account only allowed swaps 
	 */
	private void checkMultipleSwaps(int[] swapPositions) {
		MapPlace[] tmpMapPlaces = new MapPlace[numSwapPositions];
		
		float[][] somFv = new float[numSwapPositions][];
		for (int j = 0; j < numSwapPositions; j++) 
			somFv[j] = som[swapPositions[j]];
		
		for (int i = 0; i < numSwapPositions; i++) {
			
			tmpMapPlaces[i] = mapPlaces[swapPositions[i]];
			
			if (tmpMapPlaces[i] != null) {
				float[] fv = tmpMapPlaces[i].getFloatFeature();
			
				for (int j = 0; j < numSwapPositions; j++) 
					distLut[i][j] = getL2DistanceInt(fv, somFv[j]) >> 6;  // >> 4
				
					// kleinere Distanzwerte machen den Solver schneller, zu kleine Werte führen 
					// aber zu Rauschen, das wie Dithering aussieht. Achtung: die Verkleinerung  
					// muss an die typischen L2-Distanzen der Featurevektoren angepasst werden		
			}
			else {
				for (int j = 0; j < numSwapPositions; j++) 
					distLut[i][j] = 0;   
			}
		}

		int[] permutation = Solver.lapjv(distLut);

		for (int i = 0; i < numSwapPositions; i++) 
			mapPlaces[swapPositions[i]] = tmpMapPlaces[permutation[i]]; 
	}

	
	private void swapAll() {  // sehr langsam bei zu vielen FV 
		
		int[] swapPositions = new int[numSwapPositions];
		
		int k = 0;
		for (int i = 0; i < mapPlaces.length; i++) {  // count the number of fvs to swap
			if (mapPlaces[i] == null || mapPlaces[i].isSwapable()) 
				swapPositions[k++] = i;
		}
		
		if (distLut.length != numSwapPositions)
			distLut = new int[numSwapPositions][numSwapPositions];
		
		checkMultipleSwaps(swapPositions);
	}
	
	
	private final int getL2DistanceInt(final float[] fv1, final float[] fv2) {

		float dist = 0;
		for (int i = 0; i < dims; i++) {
			float d = fv1[i] - fv2[i];
			dist += d*d; 	
		}
		return (int) dist;
	}
	

	/**
	 * Internal: <br>
	 *  - writes {@link #means}, {@link #somI}, {@link #weightI}
	 * 
	 * @param radius
	 */
	private void filterSom(int radius) {
		setupIntegralImages(radius);		
		calculateWeightIntegralImage(radius);
		calculateFeatureIntegralImage(radius);
		filterSomAndNorm(radius);
	}
	
	
	/**
	 * Internal: <br>
	 *  - writes {@link #somI}, {@link #weightI}
	 */
	void setupIntegralImages(int radius) {
		
		if (oldRadius != radius) {
			nXI = nX + 2*radius + 1;
			nYI = nY + 2*radius + 1;
			weightI = new float[nXI * nYI];
			somI = new float[nXI * nYI][dims];
			oldRadius = radius;
		}
	}

	
	private void calculateWeightIntegralImage(int radius) {
		if(doWrap)
			calculateWeightIntegralImageWrap(radius);
		else
			calculateWeightIntegralImageNoWrap(radius);	
	}
	
	private void calculateWeightIntegralImageWrap(int radius) {

		for (int yI = 1; yI < nYI; yI++) {
			int y = yI - 1 - radius;
			y = (y + 2*nY) % nY;

			float sumOfTheCurrentRow = 0;
			for (int xI = 1; xI < nXI; xI++) {
				int x = xI - 1 - radius;
				x = (x + 2*nX) % nX;

				int posI = yI * nXI + xI;
				int pos  = y * nX + x;

				sumOfTheCurrentRow += weight[pos];
				weightI[posI] = weightI[posI - nXI] + sumOfTheCurrentRow;
			}
		}
	}
	
	private void calculateWeightIntegralImageNoWrap(int radius) {

		for (int yI = 1; yI < nYI; yI++) {
			int y = yI - 1 - radius;

			if (y < 0)
				y = 0;
			else if (y >= nY)
				y = nY-1;

			float sumOfTheCurrentRow = 0;
			for (int xI = 1; xI < nXI; xI++) {
				int x = xI - 1 - radius;
				if (x < 0)
					x = 0;
				else if (x >= nX)
					x = nX-1;

				int posI = yI * nXI + xI;
				int pos  = y * nX + x;

				sumOfTheCurrentRow += weight[pos];
				weightI[posI] = weightI[posI - nXI] + sumOfTheCurrentRow;
			}
		}
	}
	
	private void calculateFeatureIntegralImage(int radius) {
		if (doWrap)
			calculateFeatureIntegralImageWrap(radius);
		else
			calculateFeatureIntegralImageNoWrap(radius);	
	}
	
	private void calculateFeatureIntegralImageWrap(int radius) {

		for (int yI = 1; yI < nYI; yI++) {
			int y = yI - 1 - radius;
			y = (y + 2*nY) % nY;
			for (int xI = 1; xI < nXI; xI++) {
				int x = xI - 1 - radius;
				x = (x + 2*nX) % nX;

				int posI = yI * nXI + xI;
				int pos = y * nX + x;

				float[] a_pos = som[pos];
				float[] aI_posI       = somI[posI];
				float[] aI_posImRow   = somI[posI - nXI];
				float[] aI_posIm1     = somI[posI - 1];
				float[] aI_posImRowm1 = somI[posI - nXI - 1];

				for (int d = 0; d < dims; d++) 
					aI_posI[d] = aI_posImRow[d] + aI_posIm1[d] - aI_posImRowm1[d] + a_pos[d];
			}
		}
	}

	private void calculateFeatureIntegralImageNoWrap(int radius) {

		for (int yI = 1; yI < nYI; yI++) {
			int y = yI - 1 - radius;

			if (y < 0)
				y = 0;
			else if (y >= nY)
				y = nY-1;

			float[] sumOfTheCurrentRow = new float[dims];
			for (int xI = 1; xI < nXI; xI++) {
				int x = xI - 1 - radius;

				if (x < 0)
					x = 0;
				else if (x >= nX)
					x = nX-1;

				int posI = yI * nXI + xI;
				int pos = y * nX + x;

				float[] a_pos = som[pos];
				float[] aI_posI = somI[posI];
				float[] aI_posImRow = somI[posI - nXI];

				for (int d = 0; d < dims; d++) {
					sumOfTheCurrentRow[d] += a_pos[d];
					aI_posI[d] = aI_posImRow[d] + sumOfTheCurrentRow[d];
				}
			}
		}
	}
	
	void filterSomAndNorm(int radius) {

		for (int y1 = 0; y1 < nY; y1++) {
			int y2 = y1 + 2*radius + 1;

			for (int x1 = 0; x1 < nX; x1++) {
				int x2 = x1 + 2*radius + 1;

				int p11 = y1 * nXI + x1;
				int p12 = y1 * nXI + x2;
				int p21 = y2 * nXI + x1;
				int p22 = y2 * nXI + x2;

				float sumWeight = weightI[p22] - weightI[p21] - weightI[p12] + weightI[p11];
				float oneOverSumWeight = 1.f / sumWeight;

				float[] aI11 = somI[p11];
				float[] aI12 = somI[p12];
				float[] aI21 = somI[p21];
				float[] aI22 = somI[p22];

				int posSom = y1 * nX + x1;
				float[] actNet = som[posSom];
				
				for (int d = 0; d < dims; d++) {
					float somVal = (aI22[d] - aI21[d] - aI12[d] + aI11[d]) * oneOverSumWeight;
					actNet[d] =  somVal;
				}
			}
		}
	}
	

	

	/**
	 * Representing the input data.
	 * 
	 * @author Nico Hezel
	 */
	public static class MapPlace {
		protected int id;
		protected float[] feature;
		protected boolean isSwappable;

		public MapPlace(int id, float[] feature, boolean isSwappable) {
			this.id = id;
			this.feature = feature;
			this.isSwappable = isSwappable;
		}

		public int getId() {
			return id;
		}

		public float[] getFloatFeature() {
			return feature;
		}
		
		public boolean isSwapable() {
			return isSwappable;
		}
	}	
}