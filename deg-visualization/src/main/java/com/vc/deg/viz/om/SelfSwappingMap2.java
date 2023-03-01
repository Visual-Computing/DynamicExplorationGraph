/* (c) Copyright 2017 by K. Barthel, N. Hezel
 * All rights reserved
 */

package com.vc.deg.viz.om;


/**
 * Can not handle holes
 * 
 * @author barthel and hezel
 */
public class SelfSwappingMap2 {
	
	public static final String version = "SSM2";	// corresponds to Kai's sorterName in UniversalSorter

	private static final int actWeight = 10; 	//16;			// weight for a assigned feature vector, unassigned weight is 1
	private static final int delta = 2;      	// 2 lokale Suchweite

	// settings 
	private int swapRuns = 0;		// maximum cleanup rounds
	private int maxL = 10;			// maximum number for SSM swaps
	
	// data 
	private MapPlace[][] mapPlaces;	// [y][x]
	private short[][] inUse;		// [y][x]
	private int nX, nY, dims;
	
	// integral image 
	private int actRadius;
	private int mapPlacesXI, mapPlacesYI;
	private float[][] somI;		// Integral image of the visual SOM
	private float[] weightI;	// Integral image of the weights for the filtering
	
	// som
	private float[][] som;		
	private float[][] means; 
	private int[] weight;		// weights for the filtering
	
	public SelfSwappingMap2() { }
	
	public SelfSwappingMap2(int ssmSwapRuns, int cleanupRuns) {
		this.maxL = ssmSwapRuns;
		this.swapRuns = cleanupRuns;
	}
	
	/**
	 * Setup the SOM variables 
	 * 
	 * @param rows
	 * @param columns
	 */
	private void setupSOM(int rows, int columns, int dimensions) {
		
		// if the size of the map different than the last time, scale some variables
		if(nX != columns || nY != rows || dims != dimensions) {
			nX = columns;
			nY = rows;
			dims = dimensions;
			
			// sizes of the SOMs
			weight = new int[nX*nY];	
			som = new float[nX*nY][dims];
			means = new float[nX*nY][dims];
		}
	}
	
	/**
	 * Start the swapping process
	 * 
	 * @param map[y][x]
	 * @param inUse[y][x]
	 */
	public void run(MapPlace[][] map, short[][] inUse) {		
		setupSOM(inUse.length, inUse[0].length, map[0][0].getFloatFeature().length);
		this.inUse = inUse;
		this.mapPlaces = map;		
		runSSM();		
	}
	
	
	
	// -----------------------------------------------------------------------------------------------------------
	// -----------------------------------------------------------------------------------------------------------
	// -----------------------------------------------------------------------------------------------------------
	
	/**
	 * Run the SSM algorithms
	 */
	private void runSSM() {

		// copy the feature vectors to the SOM		
		for (int x = 0; x < nX; x++) {
			for (int y = 0; y < nY; y++) { 
				int w = (inUse[y][x] == 255) ? 1000 : 10; // höheres Gewicht für feste Bilder
				copyFeatureVectorToSom(mapPlaces[y][x].getFloatFeature(), y*nX+x, w);
			}
		}

		// filter the som
		int bs = Math.max(nX, nY); 
		filterSom(bs);
		
		do {	
			bs /= 2; 
			int rad, iter = 0;
			float step = bs/2f/maxL;
			do {
				rad = (int) Math.max(1, Math.max(bs/2,bs-iter*step));
				checkSwaps(rad, iter); 
				filterSom(rad); 
			}
			while (++iter < maxL);
		}
		while (bs > 1); 
		

		int runs = 0;
		if (swapRuns > 0) { 
			int swaps;
			do {	
				// swap places
				swaps = 0;
				for (int x = 0; x < nX; x++) 
					for (int y = 0; y < nY; y++) 
						swaps += swapPlaces(x, y);
				
				// copy features to som
				for (int x = 0; x < nX; x++) 
					for (int y = 0; y < nY; y++)
						copyFeatureVectorToSom(mapPlaces[y][x].getFloatFeature(), y*nX+x, actWeight);

				// filter som
				filterSom(1);
				//System.out.println("swaps = " + swaps);
			}
			while (swaps > 0 && runs++ < swapRuns);
		}
	}

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
	private void checkSwaps(int bs, int iter) {
		for (int y = 0; y < nY; y++)
			for (int x = 0; x < nX; x++) 
				checkSwap(bs, y, x, iter);					
	}

	/**
	 * Internal: <br>
	 * 	- write {@link #weight} <br>
	 *  - read {@link #som}, {@link #map}, {@link #inUse}, {@link #means}, {@link #perm}
	 *  
	 * @param bs
	 * @param y1
	 * @param x1
	 * @param iter
	 */
	private void checkSwap(int bs, int y1, int x1, int iter) {
		iter = 0;
		
	    x1 = (x1+iter) % nX; 
	    y1 = (y1+iter) % nY; 
	    
		int x2 = (x1+bs) % nX;
		int y2 = (y1+bs) % nY;

		int[] inuse = new int[4];
		inuse[0] = inUse[y1][x1];
		inuse[1] = inUse[y1][x2];
		inuse[2] = inUse[y2][x1];
		inuse[3] = inUse[y2][x2];

		if (inuse[0] + inuse[1] + inuse[2] + inuse[3] >= 3*255) // if 3 or 4 places are fixed, return
			return;
		
		int pos0 = y1*nX + x1;
		int pos1 = y1*nX + x2;
		int pos2 = y2*nX + x1;
		int pos3 = y2*nX + x2;

		MapPlace[] panels = new MapPlace[4];
		panels[0] = mapPlaces[y1][x1];
		panels[1] = mapPlaces[y1][x2];
		panels[2] = mapPlaces[y2][x1];
		panels[3] = mapPlaces[y2][x2];

		float[][] fv = new float[4][];
		fv[0] = panels[0].getFloatFeature();
		fv[1] = panels[1].getFloatFeature();
		fv[2] = panels[2].getFloatFeature();
		fv[3] = panels[3].getFloatFeature();

		float[][] mv = new float[4][];
		mv[0] = means[pos0];
		mv[1] = means[pos1];
		mv[2] = means[pos2];
		mv[3] = means[pos3];		

		int p = findBestPermutation(fv, mv, inuse);

		if (inuse[0] == 0) { 
			mapPlaces[y1][x1] = panels[perm[p][0]];
			copyFeatureVectorToSom(fv[perm[p][0]], pos0, actWeight);
		}
		if (inuse[1] == 0) {
			mapPlaces[y1][x2] = panels[perm[p][1]];
			copyFeatureVectorToSom(fv[perm[p][1]], pos1, actWeight);
		}
		if (inuse[2] == 0) {
			mapPlaces[y2][x1] = panels[perm[p][2]];
			copyFeatureVectorToSom(fv[perm[p][2]],pos2, actWeight);
		}
		if (inuse[3] == 0) {			
			mapPlaces[y2][x2] = panels[perm[p][3]];
			copyFeatureVectorToSom(fv[perm[p][3]], pos3, actWeight);		
		}
	}


	/** Finds the best swap (permutation) of 4 images taking into account only allowed swaps 
	 * @param fv: the four feature vectors 
	 * @param mv: the four SOM vectors
	 * @param inuse: the flags if a place is fixed
	 * @return the index of the best permutation (0 does not change anything)
	 */
	private int findBestPermutation(float[][] fv, float[][] mv, int[] inuse) {

		float[][] distLut = new float[4][4];
		for (int i = 0; i < 4; i++) {
			if (inuse[i] == 0) {
				for (int j = 0; j < 4; j++) {
					if (inuse[j] == 0)
						distLut[i][j] = getL2Distance(fv[i], mv[j]); // TODO ist hier L1 mit byte-arrays auch okey?
				}
			}
		}		

		float bestErr = distLut[0][0] + distLut[1][1] + distLut[2][2] + distLut[3][3];
		int bestPerm = 0;

		for (int i = 1; i < perm.length; i++) {	
			int[] p = perm[i];

			// only check allowed permutations
			if ( (inuse[0]==0 || (inuse[0]==255 && p[0] == 0) ) &&
				 (inuse[1]==0 || (inuse[1]==255 && p[1] == 1) ) &&
				 (inuse[2]==0 || (inuse[2]==255 && p[2] == 2) ) &&
				 (inuse[3]==0 || (inuse[3]==255 && p[3] == 3) ) ) {

				float err = 
						distLut[p[0]][0] +
						distLut[p[1]][1] +
						distLut[p[2]][2] +
						distLut[p[3]][3];	

				if (err < bestErr) {
					bestErr = err;
					bestPerm = i;
				}
			}
		}
		return bestPerm;
	}

	private static final int[][] perm = {
			{ 0, 1, 2, 3}, // 0
			{ 0, 1, 3, 2}, // 1
			{ 0, 2, 1, 3}, // 2
			{ 0, 2, 3, 1}, // 3
			{ 0, 3, 1, 2}, // 4
			{ 0, 3, 2, 1}, // 5
			{ 1, 0, 2, 3}, // 6
			{ 1, 0, 3, 2}, // 7
			{ 1, 2, 0, 3}, // 8
			{ 1, 2, 3, 0}, // 9
			{ 1, 3, 0, 2}, // 10
			{ 1, 3, 2, 0}, // 11
			{ 2, 0, 1, 3}, // 12
			{ 2, 0, 3, 1}, // 13
			{ 2, 1, 0, 3}, // 14
			{ 2, 1, 3, 0}, // 15
			{ 2, 3, 0, 1}, // 16
			{ 2, 3, 1, 0}, // 17
			{ 3, 0, 1, 2}, // 18
			{ 3, 0, 2, 1}, // 19
			{ 3, 1, 0, 2}, // 20
			{ 3, 1, 2, 0}, // 21
			{ 3, 2, 0, 1}, // 22
			{ 3, 2, 1, 0}, // 23
	};

	private float getL2Distance(float[] fv1, float[] fv2) {
		float dist = 0;
		for (int i = 0; i < fv1.length; i++) {
			float d = fv1[i] - fv2[i];
			dist += d*d;
		}
		return dist;
	}

	/**
	 * Internal: <br>
	 * 	- write {@link #weight} <br>
	 *  - read {@link #som}
	 *  
	 * @param visSearchFV
	 * @param pos
	 * @param w
	 */
	private void copyFeatureVectorToSom(float[] visSearchFV, int pos, int w) {
		// copy the weighted feature vector to the SOM (which will be filtered next)
		float[] actVisSom = som[pos];
		for (int d = 0; d < dims; d++) 	
			actVisSom[d] = visSearchFV[d] * w;
		
		// change the weight for this position
		weight[pos] = w;
	}


	/**
	 * Internal: <br>
	 *  - writes {@link #means}, {@link #somI}, {@link #weightI}
	 * 
	 * @param rad
	 */
	private void filterSom(int rad) {
		actRadius = rad;
		setupIntegralImages();		
		calculateIntegralImageNoWrap();
		filterSom();
	}

	/**
	 * Internal: <br>
	 *  - writes {@link #somI}, {@link #weightI}
	 */
	private void setupIntegralImages() {
		mapPlacesXI = nX + 2*actRadius + 1;
		mapPlacesYI = nY + 2*actRadius + 1;
		weightI = new float[mapPlacesXI * mapPlacesYI];
		somI = new float[mapPlacesXI * mapPlacesYI][dims];
	}



	/**
	 * Internal: <br>
	 * 	- write {@link #weightI}, {@link #somI}<br>
	 *  - read {@link #weight}, {@link #som}
	 */
	private void calculateIntegralImageNoWrap() {

		for (int yI = 1; yI < mapPlacesYI; yI++) {
			int y = yI - 1 - actRadius;
			if (y < 0)
				y = -y; 
			else if (y >= nY)
				y = 2*nY-y-1;

			if (y < 0)
				y = 0;
			if (y >= nY)
				y = nY-1;

			float weightSumOfTheCurrentRow = 0;
			float[] somSumOfTheCurrentRow = new float[dims];
			for (int xI = 1; xI < mapPlacesXI; xI++) {
				int x = xI - 1 - actRadius;
				if (x < 0)
					x = -x; 
				else if (x >= nX)
					x = 2*nX-x-1;
				if (x < 0)
					x = 0;
				if (x >= nX)
					x = nX-1;

				int posI = yI * mapPlacesXI + xI;
				int pos  = y * nX + x;

				// weight values
				weightSumOfTheCurrentRow += weight[pos];
				weightI[posI] = weightI[posI - mapPlacesXI] + weightSumOfTheCurrentRow;
				
				// SOM values
				float[] a_pos = som[pos];
				float[] aI_posI = somI[posI];
				float[] aI_posImRow = somI[posI - mapPlacesXI];

				for (int d = 0; d < dims; d++) {
					somSumOfTheCurrentRow[d] += a_pos[d];
					aI_posI[d] = aI_posImRow[d] + somSumOfTheCurrentRow[d];
				}
			}
		}
	}
	

	
	/**
	 * Internal: <br>
	 * 	- writes {@link #means} <br>
	 *  - reads {@link #somI}, {@link #weightI}
	 */
	private void filterSom() {

		for (int y1 = 0; y1 < nY; y1++) {
			final int y2 = y1 + 2*actRadius + 1;

			for (int x1 = 0; x1 < nX; x1++) {
				final int x2 = x1 + 2*actRadius + 1;

				final int p11 = y1 * mapPlacesXI + x1;
				final int p12 = y1 * mapPlacesXI + x2;
				final int p21 = y2 * mapPlacesXI + x1;
				final int p22 = y2 * mapPlacesXI + x2;

				final float[] aI11 = somI[p11];
				final float[] aI12 = somI[p12];
				final float[] aI21 = somI[p21];
				final float[] aI22 = somI[p22];

				final float sumWeight = weightI[p22] - weightI[p21] - weightI[p12] + weightI[p11];
				final float oneOverSumWeight = 1.f / sumWeight;

				final int posSom = y1 * nX + x1;


				float[] actNet = means[posSom];
				for (int d = 0; d < dims; d++)
					actNet[d] = (aI22[d] - aI21[d] - aI12[d] + aI11[d]) * oneOverSumWeight;
			}
		}
	}

	private int swapPlaces(int x1, int y1) {

		if (inUse[y1][x1] == 255)
			return 0;
		
		MapPlace panel1 = mapPlaces[y1][x1];
		int id1 = panel1.getId();
		float dist_pos1_id1 = getNeighborDistance(x1, y1);

		// full search for the best place
		int bestX = -1, bestY = -1;
		{
			int bestI = -1;
			float bestDist = Float.MAX_VALUE;
			float[] featureVector = panel1.getFloatFeature();
			for (int i = 0; i < nX*nY; i++) { 
				float dist = getL2Distance(featureVector, som[i]);
				if (dist < bestDist) {
					bestDist = dist;
					bestI = i;
				}
			}
			bestX = bestI % nX;
			bestY = bestI / nX;
		}

		for (int x = bestX - delta; x < bestX + delta + 2; x++) {		
			int x2 = (x +  nX) % nX;
			for (int y = bestY - delta; y < bestY + delta + 2; y++) {
				int y2 = (y +  nY) % nY;
				
				if (inUse[y2][x2] == 0) {

					MapPlace panel2 = mapPlaces[y2][x2];
					if (id1 != panel2.getId()) {
						float dist_pos2_id2 = getNeighborDistance(x2, y2);

						// temp swap
						mapPlaces[y1][x1] = panel2;
						mapPlaces[y2][x2] = panel1;

						float dist_pos2_id1 = getNeighborDistance(x2, y2);
						float dist_pos1_id2 = getNeighborDistance(x1, y1);

						float gain = (dist_pos1_id1 + dist_pos2_id2) - (dist_pos2_id1 + dist_pos1_id2);

						if (gain > 0) {
							//System.out.println("Gain: " + gain);
							return 1;
						}
						else { 	// swap back
							mapPlaces[y1][x1] = panel1;
							mapPlaces[y2][x2] = panel2;
						}		
					}
				}
			}
		}
		return 0;
	}
	
	private float getNeighborDistance(int x1, int y1) {  

		float[] featureVector1 = mapPlaces[y1][x1].getFloatFeature();
		int numNeighbors = 4;
		float distTotal = 0;
		int count = 0;

		int x = 0, y = 0;
		int dx = 0, dy = -1;
		do {
			if (y == 0 && x >= 0) {
				dx = 0; dy = -1;
			}
			else if (y == 0)
				dx = -dx; 
			if (y == -1) 
				dx = -1;
			if (x == 0 && y != 0)
				dy = -dy;

			x += dx;
			y += dy;

			int y2 = y1 + y; 
			int x2 = x1 + x;
			if (x2 >= 0 && x2 < nX && y2 >= 0 && y2 < nY) {
				float[] featureVector2 = mapPlaces[y2][x2].getFloatFeature();
				distTotal += getL2Distance(featureVector1, featureVector2);
				count += 1;
			}
		}
		while (count < numNeighbors);
		
		return distTotal;
	}

	
	/**
	 * Representing the input data.
	 * 
	 * @author Nico Hezel
	 */
	public static class MapPlace {
		protected int id;
		protected float[] feature;

		public MapPlace(int id, float[] feature) {
			this.id = id;
			this.feature = feature;
		}

		public int getId() {
			return id;
		}

		public float[] getFloatFeature() {
			return feature;
		}
	}	
}