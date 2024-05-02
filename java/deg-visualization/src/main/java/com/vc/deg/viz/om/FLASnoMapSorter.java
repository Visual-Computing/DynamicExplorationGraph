package com.vc.deg.viz.om;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;

/*
 * Fast Linear Assignment Sorting
 * 
 * Can handle holes and fixed cells. 
 */
public class FLASnoMapSorter {

	public static int   QUANT = 256; // /256;  quantized distance steps

	// hyper parameter 
	public static int   MaxSwapPositions = 9;
	public static float SampleFactor = 1.0f;	// 1 the fraction of swaps per iteration
	public static float RadiusDecay = 0.99f;
	public static float EndRadius = 1.0f; 
	public static float InitialRadiusFactor = 0.5f;

	protected final int maxSwapPositions;
	protected final float sampleFactor;
	protected final float radiusDecay;
	protected final float radiusEnd;
	protected final float initialRadFactor;

	protected final float weightHole = 0.01f;    	// adjust to the amount of holes
	protected final float weightSwappable = 1f;  
	protected final float weightNonSwappable = 100f;  

	protected final Random random;
    
	private int rows = 0;
	private int columns = 0;
	
	private float[][] distMatrix;
	private boolean hasFixedPositions = false;
	private byte[] positionStatus = null;  // indicates the state of the position (swappable=0, border=1, fixed=2)
	private int[] borderPositions = null;
	private static final int FIXED  = 2;
	private static final int BORDER = 1;
    
    // temporary variables
    private int[] swapPositions;
    private MapPlace[] swapCandidates;
    private int[][] swapDistLut;
    private float[][] swapDistLutF;
    
    private float[][] invGeoDist0;
    private float[][] featureDist0;


	public FLASnoMapSorter(Random random) {
		this.random = random;
		this.sampleFactor = SampleFactor;	
		this.maxSwapPositions = MaxSwapPositions;
		this.radiusDecay = RadiusDecay;
		this.radiusEnd = EndRadius;
		this.initialRadFactor = InitialRadiusFactor;
	}
	

	/**
	 * Precompute all the distances
	 * 
	 * @param grid
	 * @return
	 */
	public static float[][] precomputedDistFunc(MapPlace[] grid, FeatureSpace distFunc, byte[] positionStatus) {
		final List<MapPlace> elements = new ArrayList<>();
		for (int i = 0; i < grid.length; i++) {
			final MapPlace element = grid[i];
			if(element.getId() != -1) {
				element.interalIndex = elements.size(); // all none hole cells get an internal index
				elements.add(element);
			}
		}
		
		final int elementCount = elements.size();
		final MapPlace[] sortedElements = new MapPlace[elementCount];
		for (int i = 0; i < elementCount; i++) {
			final MapPlace element = elements.get(i);
			sortedElements[element.interalIndex] = element;
		}
		
		final float[][] distMatrix = new float[elementCount][elementCount];
		for (int i = 0; i < elementCount-1; i++) 
			for (int j = i + 1; j < elementCount; j++) 
				if (positionStatus[i] + positionStatus[j] <= BORDER)
					distMatrix[i][j] = distMatrix[j][i] = distFunc.computeDistance(sortedElements[i].feature, sortedElements[j].feature);
		
		return distMatrix;
	}
	

	public void doSorting(MapPlace[] imageGrid, int columns, int rows, FeatureSpace distFunc) {
		this.columns = columns;
		this.rows = rows;

		// prepare auxiliary data for fixed positions
		this.hasFixedPositions = Arrays.stream(imageGrid).anyMatch(MapPlace::isFixed);
		this.positionStatus = new byte[imageGrid.length]; // prefilled with status SWAPPABLE		
		if(hasFixedPositions) {

			// mark all fixed positions
			final boolean[] isFixedPosition = new boolean[imageGrid.length];
			for (int i = 0; i < imageGrid.length; i++) 
				isFixedPosition[i] = imageGrid[i].isFixed;

			// gather all positions of the fixed elements 			
			int fixedPosCount = 0;
			final int[] fixedPositionCheck = new int[imageGrid.length];				
			for (int i = 0; i < isFixedPosition.length; i++) { 
				if(isFixedPosition[i]) {
					byte status = FIXED;

					// check if surrounded by other fiexed elements
					boolean check = false;

					// north 
					int j = i - columns;	
					if (j >= 0 && !isFixedPosition[j])
						check = true;

					// south 
					j = i + columns;	
					if (j < columns*rows && !isFixedPosition[j])
						check = true;

					// east 
					j = i + 1;	
					if (j / columns == i / columns  && !isFixedPosition[j])
						check = true;

					// west 
					j = i - 1;	
					if (j >= 0 && j / columns == i / columns  && !isFixedPosition[j])
						check = true;

					if (check) {
						status = BORDER;
						fixedPositionCheck[fixedPosCount++] = i;
					}

					positionStatus[i] = status;
				}
			}
			this.borderPositions = Arrays.copyOf(fixedPositionCheck, fixedPosCount);
		}		
		this.distMatrix = precomputedDistFunc(imageGrid, distFunc, positionStatus);
		
		// temporary variables using the maximal swap position count
		this.swapPositions = new int[Math.min(maxSwapPositions, rows*columns)];
		this.swapCandidates = Arrays.copyOf(imageGrid, swapPositions.length);
		this.swapDistLut  = new int[swapPositions.length][swapPositions.length];
		this.swapDistLutF  = new float[swapPositions.length][swapPositions.length];
		
		if(borderPositions != null) {
			this.invGeoDist0 = new float[swapCandidates.length][borderPositions.length];
			this.featureDist0 = new float[swapCandidates.length][borderPositions.length];
		}
		
		// setup the initial radius
		float rad = Math.max(columns, rows)*initialRadFactor;	
			
		do {
			int radius = (int) Math.max(1, Math.round(rad));  // set the radius			
			checkRandomSwaps(radius, imageGrid); 
			rad *= radiusDecay;  
		}
		while (rad > radiusEnd); 
	}

	
	
	
	// -------------------------------------------------------------------------------------------------------------
	// ---------------------------------------- Swapping and Solver part--------------------------------------------
	// -------------------------------------------------------------------------------------------------------------
	
	private static void shuffleArray(int[] array, Random random)
	{
		int index, temp;
		for (int i = array.length - 1; i > 0; i--)
		{
			index = random.nextInt(i + 1);
			temp = array[index];
			array[index] = array[i];
			array[i] = temp;
		}
	}
	
	private void checkRandomSwaps(int radius, MapPlace[] imageGrid) {

		// set swap size
		int swapAreaWidth = Math.min(2*radius+1, columns);
		int swapAreaHeight = Math.min(2*radius+1, rows);
		int k = 0;
		while (swapAreaHeight * swapAreaWidth < swapPositions.length) {
			if (k++ % 2 == 0) // alternate the size increase
				swapAreaWidth = Math.min(swapAreaWidth+1, columns);
			else
				swapAreaHeight = Math.min(swapAreaHeight+1, rows);
		}	

		// get all positions of the actual swap region
		final int[] swapIndices = new int[swapAreaWidth*swapAreaHeight];
		for (int i = 0, y = 0; y < swapAreaHeight; y++)
			for (int x = 0; x < swapAreaWidth; x++)
				swapIndices[i++] = y*columns + x;
		
		if (swapAreaHeight * swapAreaWidth > swapPositions.length)
			shuffleArray(swapIndices, random);

		final int numSwapTries = (int) Math.max(1, (sampleFactor * rows * columns / swapPositions.length));
		for (int n = 0; n < numSwapTries; n++) {			
			final int numSwapPositions = findSwappablePositions(swapIndices, swapPositions, swapAreaWidth, swapAreaHeight); 
			doSwaps(swapPositions, numSwapPositions, imageGrid, radius);
		}	
	}
	

	private int findSwappablePositions(int[] swapIndices, int[] swapPositions, int swapAreaWidth, int swapAreaHeight) {
		
		// calculate start position of swap area
		int pos0 = random.nextInt(rows*columns);
		final int x0 =  pos0 % columns;
		final int y0 =  pos0 / columns;
		
		int xStart = Math.max(0, x0 - swapAreaWidth/2);     
		if (xStart + swapAreaWidth > columns)
			xStart = columns-swapAreaWidth;

		int yStart = Math.max(0, y0 - swapAreaHeight/2); 
		if (yStart + swapAreaHeight > rows)
			yStart = rows-swapAreaHeight;
		
		pos0 = yStart * columns + xStart;
		
		final int startIndex = (swapIndices.length - swapPositions.length > 0) ? random.nextInt(swapIndices.length - swapPositions.length) : 0;
		int numSwapPositions = 0;
		for (int j = startIndex; j < swapIndices.length && numSwapPositions < swapPositions.length; j++) {

			int pos = (pos0 + swapIndices[j]) % (rows * columns); 
			
			if (hasFixedPositions == false || positionStatus[pos] == 0) 
				swapPositions[numSwapPositions++] = pos;
		}
	
		return numSwapPositions;
	}	

	private void doSwaps(int[] swapPositions, int numSwapPositions, MapPlace[] imageGrid, int radius) { 
		
		for (int i = 0; i < numSwapPositions; i++) 
			swapCandidates[i] = imageGrid[swapPositions[i]];		
		
		swapDistLut = calcDistLut(swapCandidates, swapPositions, numSwapPositions, imageGrid, radius);
	   
		int[] permutation = JonkerVolgenantSolver.computeAssignment(swapDistLut, numSwapPositions);	
		for (int i = 0; i < numSwapPositions; i++)
			imageGrid[swapPositions[permutation[i]]] = swapCandidates[i];
	}
	
	
	private int[][] calcDistLut(MapPlace[] swapCandidates, int[] swapCandidatesPos, int numSwapCandidates, MapPlace[] imageGrid, int rad) {
		
		final boolean testFixedElements = hasFixedPositions && random.nextInt(10) == 0; // only 10% of the time
		if(testFixedElements) {
			for (int i = 0; i < numSwapCandidates; i++) {
				final int xi = swapCandidatesPos[i] % columns;
				final int yi = swapCandidatesPos[i] / columns;
				final MapPlace swapCandidate = swapCandidates[i];
				
				for (int f = 0; f < borderPositions.length; f++) {
					final int pos0 = borderPositions[f];
					final int x0 = pos0 % columns;
					final int y0 = pos0 / columns;
					final MapPlace pos0Element = imageGrid[pos0];
					
					if (pos0Element.getId() != -1 && swapCandidate.getId() != -1) {
						invGeoDist0[i][f] = 100f / ((x0 - xi) * (x0 - xi) + (y0 - yi) * (y0 - yi)); 
						featureDist0[i][f] = distMatrix[swapCandidate.interalIndex][pos0Element.interalIndex];
					}
				}
			}
		}
				
		float max = 0;
		final int delta = random.nextInt(rad) + 1;		

		for (int j = 0; j < numSwapCandidates; j++) {       // positions

			int xj = swapCandidatesPos[j] % columns;
			int yj = swapCandidatesPos[j] / columns;
			
			int xjm = Math.max(xj - delta, 0);
			int xjp = Math.min(xj + delta, columns-1);
			int yjm = Math.max(yj - delta, 0);
			int yjp = Math.min(yj + delta, rows-1);
			
			for (int i = 0; i < numSwapCandidates; i++) {           // elements
				float val = 0;
				int counter = 0;
				
				// holes have a distance of 0 to all other elements
				if(swapCandidates[i].getId() != -1) {
					final float[] candidateDistLUT = distMatrix[swapCandidates[i].interalIndex];
					
					int pos;
					pos = yj  * columns + xjm;  // West
					if (positionStatus[pos] <= BORDER && imageGrid[pos].getId() != -1) {
						val += candidateDistLUT[imageGrid[pos].interalIndex];
						counter++;
					}
					pos = yj  * columns + xjp;  // East
					if (positionStatus[pos] <= BORDER && imageGrid[pos].getId() != -1) {
						val += candidateDistLUT[imageGrid[pos].interalIndex];
						counter++;
					}
					pos = yjm * columns + xj;   // North
					if (positionStatus[pos] <= BORDER && imageGrid[pos].getId() != -1) {
						val += candidateDistLUT[imageGrid[pos].interalIndex];
						counter++;
					}
					pos = yjp * columns + xj;   // South
					if (positionStatus[pos] <= BORDER && imageGrid[pos].getId() != -1) {
						val += candidateDistLUT[imageGrid[pos].interalIndex];
						counter++;
					}
					pos = yjm * columns + xjm;  // North West
					if (positionStatus[pos] <= BORDER && imageGrid[pos].getId() != -1) {
						val += candidateDistLUT[imageGrid[pos].interalIndex];
						counter++;
					}
					pos = yjm * columns + xjp;  // North East
					if (positionStatus[pos] <= BORDER && imageGrid[pos].getId() != -1) {
						val += candidateDistLUT[imageGrid[pos].interalIndex];
						counter++;
					}
					pos = yjp * columns + xjm;  // South West
					if (positionStatus[pos] <= BORDER && imageGrid[pos].getId() != -1) {
						val += candidateDistLUT[imageGrid[pos].interalIndex];
						counter++;
					}
					pos = yjp * columns + xjp;  // South East 
					if (positionStatus[pos] <= BORDER && imageGrid[pos].getId() != -1) {
						val += candidateDistLUT[imageGrid[pos].interalIndex];
						counter++;
					}
					val = 8 * val / counter;
	
					// additional costs
					if(testFixedElements)
						for (int f = 0; f < borderPositions.length; f++) 
							val += featureDist0[i][f] * invGeoDist0[j][f];
				}
				
				swapDistLutF[i][j] = val;
				if (val > max)
					max = val;
			}
		}
		
		for (int i = 0; i < numSwapCandidates; i++) 
			for (int j = 0; j < numSwapCandidates; j++) 
				swapDistLut[i][j] = (int) (QUANT * swapDistLutF[i][j] / max + 0.5);		
				
		return swapDistLut;
	}	
	
	@Override
	public String toString() {
		return "FLASnoMap";
	}


	/**
	 * Representing the input data.
	 * 
	 * @author Nico Hezel
	 */
	public static class MapPlace {
		private final int id;
		private final FeatureVector feature;
		private final boolean isFixed;
		
		private int interalIndex = -1;

		public MapPlace(int id, FeatureVector feature, boolean isFixed) {
			this.id = id;
			this.feature = feature;
			this.isFixed = isFixed;
		}
		
		public int getId() {
			return id;
		}
		
		public boolean isFixed() {
			return isFixed;
		}
	}
}