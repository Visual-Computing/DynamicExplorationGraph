package com.vc.deg.viz.om;

import java.util.Arrays;
import java.util.Random;

/*
 * Fast Linear Assignment Sorting
 * 
 * Can handle holes and fixed cells. 
 */
public class FastLinearAssignmentSorter {

	public static int   QUANT = 256; // /256;  quantized distance steps

	// hyper parameter 
	public static int   MaxSwapPositions = 9;
	public static float SampleFactor = 1.0f;	// 1 the fraction of swaps per iteration
	public static float RadiusDecay = 0.93f;
	public static float EndRadius = 1.0f; 
	public static float InitialRadiusFactor = 0.5f;
	public static int   NumFilters = 1;

	protected final int maxSwapPositions;
	protected final float sampleFactor;
	protected final float radiusDecay;
	protected final float radiusEnd;
	protected final float initialRadFactor;
	protected final int numFilters;

	protected final float weightHole = 0.01f;    	// TODO adjust to the amount of holes
	protected final float weightSwappable = 1f;  
	protected final float weightNonSwappable = 100f;  

	protected final Random random;
	protected final boolean doWrap;



	private int dim = -1; 
	private int rows = 0;
	private int columns = 0;

	private float[][] som;
	private float[] weights;    

	// temporary variables
	private int[] swapPositions;
	private float[][] fvs;
	private float[][] somFvs;
	private MapPlace[] swapedElements;
	private int[][] distLut;
	private float[][] distLutF;

	public FastLinearAssignmentSorter(Random random, boolean doWrap) {
		this.random = random;
		this.doWrap = doWrap;
		this.sampleFactor = SampleFactor;								// 1.0
		this.maxSwapPositions = MaxSwapPositions;
		this.radiusDecay = RadiusDecay;  								// 0.6
		this.radiusEnd = EndRadius;								        // 1.0
		this.initialRadFactor = InitialRadiusFactor;  					// 0.5
		this.numFilters = NumFilters;
	}

	public void doSorting(MapPlace[] imageGrid, int columns, int rows) {		
		this.columns = columns;
		this.rows = rows;

		final int gridSize = imageGrid.length;
		for (int i = 0; i < gridSize; i++) {  // get dimension of fv
			if (imageGrid[i] != null) {
				dim = imageGrid[i].getFloatFeature().length;	
				break;
			}
		}

		this.som = new float[gridSize][dim];
		this.weights = new float[gridSize];

		// temporary variables using the maximal swap position count
		this.swapPositions = new int[Math.min(maxSwapPositions, rows*columns)];
		this.fvs = new float[swapPositions.length][];
		this.somFvs = new float[swapPositions.length][];
		this.swapedElements = Arrays.copyOf(imageGrid, swapPositions.length);
		this.distLut  = new int[swapPositions.length][swapPositions.length];
		this.distLutF  = new float[swapPositions.length][swapPositions.length];



		// setup the initial radius
		float rad = Math.max(columns, rows)*initialRadFactor;	

		// try to improve the map
		do {
			copyFeatureVectorsToSom(imageGrid);

			final int radius = (int) Math.max(1, Math.round(rad));  // set the radius
			final int radiusX = Math.max(1, Math.min(columns/2, radius));
			final int radiusY = Math.max(1, Math.min(rows/2, radius));
			rad *= radiusDecay;  

			for (int i = 0; i < numFilters; i++) 
				filterWeightedSom(radiusX, radiusY, columns, rows, som, dim, weights, doWrap);

			checkRandomSwaps(radius, imageGrid, som); 
		}
		while (rad > radiusEnd); 
	}

	private void copyFeatureVectorsToSom(MapPlace[] imageGrid) {
		for (int pos = 0; pos < imageGrid.length; pos++)  {

			final float[] somCell = som[pos];
			final MapPlace cell = imageGrid[pos];

			// handle holes
			if (cell != null) {
				final float[] fv = cell.getFloatFeature();

				// higher weight for fixed images
				float w = cell.isSwapable() ? weightSwappable : weightNonSwappable; 
				for (int i = 0; i < fv.length; i++) 
					somCell[i] = w * fv[i];
				weights[pos] = w; 
			}
			else {
				for (int i = 0; i < dim; i++) 
					somCell[i] *= weightHole;
				weights[pos] = weightHole;
			}
		}
	}

	protected static void filterWeightedSom(int actRadiusX, int actRadiusY, int columns, int rows, float[][] som, int dim, float[] weights, boolean doWrap) {

		int filterSizeX = 2*actRadiusX+1;
		int filterSizeY = 2*actRadiusY+1;

		float[][] somH = new float[rows * columns][dim];
		float[] weightsH = new float[rows * columns];

		if(doWrap) {
			filterHwrap(som, somH, rows, columns, dim, filterSizeX);
			filterHwrap(weights, weightsH, rows, columns, filterSizeX);

			filterVwrap(somH, som, rows, columns, dim, filterSizeY);	
			filterVwrap(weightsH, weights, rows, columns, filterSizeY);	

		}
		else {
			filterHmirror(som, somH, rows, columns, dim, filterSizeX);
			filterHmirror(weights, weightsH, rows, columns, filterSizeX);

			filterVmirror(somH, som, rows, columns, dim, filterSizeY);	
			filterVmirror(weightsH, weights, rows, columns, filterSizeY);	
		}	

		for (int i = 0; i < som.length; i++) {
			float w = 1 / weights[i];
			for (int d = 0; d < dim; d++) 
				som[i][d] *= w;		
		}
	}


	protected static void filterVmirror(float[][] input, float[][] output, int rows, int columns, int dims, int filterSize) {

		int ext = filterSize/2;		// size of the border extension

		float[][] colExt = new float[rows + 2*ext][];  // extended row

		// filter the columns
		for (int x = 0; x < columns; x++) {

			for (int i = 0; i < rows; i++) 
				colExt[i+ext] = input[x + i*columns]; // copy one column 

			// mirrored extension
			for (int i = 0; i < ext; i++) {
				colExt[ext-1-i] = colExt[ext+i+1];
				colExt[rows + ext + i] = colExt[ext+rows-2-i];
			}

			float[] tmp = new float[dims]; 
			for (int i = 0; i < filterSize; i++) // first element
				for (int d = 0; d < dims; d++) 
					tmp[d] += colExt[i][d];

			for (int d = 0; d < dims; d++) 
				output[x][d] = tmp[d] / filterSize;

			for (int i = 1; i < rows; i++) { // rest of the column
				int left = i-1;
				int right = left + filterSize;

				for (int d = 0; d < dims; d++) { 
					tmp[d] += colExt[right][d] - colExt[left][d];
					output[x + i*columns][d] = tmp[d] / filterSize; 
				}
			}
		}
	}

	protected static void filterVmirror(float[] input, float[] output, int rows, int columns, int filterSize) {

		int ext = filterSize/2;		// size of the border extension

		float[] colExt = new float[rows + 2*ext];  // extended row

		// filter the columns
		for (int x = 0; x < columns; x++) {

			for (int i = 0; i < rows; i++) 
				colExt[i+ext] = input[x + i*columns]; // copy one column 

			// mirrored extension
			for (int i = 0; i < ext; i++) {
				colExt[ext-1-i] = colExt[ext+i+1];
				colExt[rows + ext + i] = colExt[ext+rows-2-i];
			}

			float tmp = 0; 
			for (int i = 0; i < filterSize; i++) // first element
				tmp += colExt[i];

			output[x] = tmp / filterSize;

			for (int i = 1; i < rows; i++) { // rest of the column
				int left = i-1;
				int right = left + filterSize;

				tmp += colExt[right] - colExt[left];
				output[x + i*columns] = tmp / filterSize; 
			}
		}
	}


	protected static void filterHmirror(float[][] input, float[][] output, int rows, int columns, int dims, int filterSize) {

		int ext = filterSize/2;							  // size of the border extension

		float[][] rowExt = new float[columns + 2*ext][];  // extended row

		// filter the rows
		for (int y = 0; y < rows; y++) {

			int actRow = y*columns;

			for (int i = 0; i < columns; i++) 
				rowExt[i+ext] = input[actRow + i]; // copy one row 

			// mirrored extension
			for (int i = 0; i < ext; i++) {
				rowExt[ext-1-i] = rowExt[ext+i+1];
				rowExt[columns + ext+i] = rowExt[columns + ext -2 -i];
			}

			float[] tmp = new float[dims]; 
			for (int i = 0; i < filterSize; i++) // first element
				for (int d = 0; d < dims; d++) 
					tmp[d] += rowExt[i][d];

			for (int d = 0; d < dims; d++) 
				output[actRow][d] = tmp[d] / filterSize;

			for (int i = 1; i < columns; i++) { // rest of the row
				int left = i-1;
				int right = left + filterSize;

				for (int d = 0; d < dims; d++) { 
					tmp[d] += rowExt[right][d] - rowExt[left][d];
					output[actRow + i][d] = tmp[d] / filterSize; 
				}
			}
		}
	}

	protected static void filterHmirror(float[] input, float[] output, int rows, int columns, int filterSize) {

		int ext = filterSize/2;							  // size of the border extension

		float[] rowExt = new float[columns + 2*ext];  // extended row

		// filter the rows
		for (int y = 0; y < rows; y++) {

			int actRow = y*columns;

			for (int i = 0; i < columns; i++) 
				rowExt[i+ext] = input[actRow + i]; // copy one row 

			// mirrored extension
			for (int i = 0; i < ext; i++) {
				rowExt[ext-1-i] = rowExt[ext+i+1];
				rowExt[columns + ext+i] = rowExt[columns + ext -2 -i];
			}

			float tmp = 0; 
			for (int i = 0; i < filterSize; i++) // first element
				tmp += rowExt[i];

			output[actRow] = tmp / filterSize;

			for (int i = 1; i < columns; i++) { // rest of the row
				int left = i-1;
				int right = left + filterSize;

				tmp += rowExt[right] - rowExt[left];
				output[actRow + i] = tmp / filterSize; 
			}
		}
	}

	protected static void filterVwrap(float[][] input, float[][] output, int rows, int columns, int dims, int filterSize) {

		int ext = filterSize/2;		// size of the border extension

		float[][] colExt = new float[rows + 2*ext][];  // extended row

		// filter the columns
		for (int x = 0; x < columns; x++) {

			for (int i = 0; i < rows; i++) 
				colExt[i+ext] = input[x + i*columns]; // copy one column 

			// wrapped extension
			for (int i = 0; i < ext; i++) {
				colExt[ext-1-i] = colExt[rows+ext-i-1];
				colExt[rows+ext+i] = colExt[ext+i];
			}

			float[] tmp = new float[dims]; 
			for (int i = 0; i < filterSize; i++) // first element
				for (int d = 0; d < dims; d++) 
					tmp[d] += colExt[i][d];

			for (int d = 0; d < dims; d++) 
				output[x][d] = tmp[d] / filterSize;

			for (int i = 1; i < rows; i++) { // rest of the column
				int left = i-1;
				int right = left + filterSize;

				for (int d = 0; d < dims; d++) { 
					tmp[d] += colExt[right][d] - colExt[left][d];
					output[x + i*columns][d] = tmp[d] / filterSize; 
				}
			}
		}
	}

	protected static void filterVwrap(float[] input, float[] output, int rows, int columns, int filterSize) {

		int ext = filterSize/2;		// size of the border extension

		float[] colExt = new float[rows + 2*ext];  // extended row

		// filter the columns
		for (int x = 0; x < columns; x++) {

			for (int i = 0; i < rows; i++) 
				colExt[i+ext] = input[x + i*columns]; // copy one column 

			// wrapped extension
			for (int i = 0; i < ext; i++) {
				colExt[ext-1-i] = colExt[rows+ext-i-1];
				colExt[rows+ext+i] = colExt[ext+i];
			}

			float tmp = 0; 
			for (int i = 0; i < filterSize; i++) // first element
				tmp += colExt[i];

			output[x] = tmp / filterSize;

			for (int i = 1; i < rows; i++) { // rest of the column
				int left = i-1;
				int right = left + filterSize;

				tmp += colExt[right] - colExt[left];
				output[x + i*columns] = tmp / filterSize; 
			}
		}
	}

	protected static void filterHwrap(float[][] input, float[][] output, int rows, int columns, int dims, int filterSize) {

		int ext = filterSize/2;							  // size of the border extension

		float[][] rowExt = new float[columns + 2*ext][];  // extended row

		// filter the rows
		for (int y = 0; y < rows; y++) {

			int actRow = y*columns;

			for (int i = 0; i < columns; i++) 
				rowExt[i+ext] = input[actRow + i]; // copy one row 

			// wrapped extension
			for (int i = 0; i < ext; i++) {
				rowExt[ext-1-i] = rowExt[columns+ext-i-1];
				rowExt[columns+ext+i] = rowExt[ext+i];
			}

			float[] tmp = new float[dims]; 
			for (int i = 0; i < filterSize; i++) // first element
				for (int d = 0; d < dims; d++) 
					tmp[d] += rowExt[i][d];

			for (int d = 0; d < dims; d++) 
				output[actRow][d] = tmp[d] / filterSize;

			for (int i = 1; i < columns; i++) { // rest of the row
				int left = i-1;
				int right = left + filterSize;

				for (int d = 0; d < dims; d++) { 
					tmp[d] += rowExt[right][d] - rowExt[left][d];
					output[actRow + i][d] = tmp[d] / filterSize; 
				}
			}
		}
	}

	protected static void filterHwrap(float[] input, float[] output, int rows, int columns, int filterSize) {

		int ext = filterSize/2;							  // size of the border extension

		float[] rowExt = new float[columns + 2*ext];  // extended row

		// filter the rows
		for (int y = 0; y < rows; y++) {

			int actRow = y*columns;

			for (int i = 0; i < columns; i++) 
				rowExt[i+ext] = input[actRow + i]; // copy one row 

			// wrapped extension
			for (int i = 0; i < ext; i++) {
				rowExt[ext-1-i] = rowExt[columns+ext-i-1];
				rowExt[columns+ext+i] = rowExt[ext+i];
			}

			float tmp = 0; 
			for (int i = 0; i < filterSize; i++) // first element
				tmp += rowExt[i];

			output[actRow] = tmp / filterSize;

			for (int i = 1; i < columns; i++) { // rest of the row
				int left = i-1;
				int right = left + filterSize;
				tmp += rowExt[right] - rowExt[left];
				output[actRow + i] = tmp / filterSize; 
			}
		}
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

	private void checkRandomSwaps(int radius, MapPlace[] imageGrid, float[][] som) {

		// set swap size
		int swapAreaWidth = Math.min(2*radius+1, columns);
		int swapAreaHeight = Math.min(2*radius+1, rows);
		while (swapAreaHeight * swapAreaWidth < swapPositions.length) {
			if (swapAreaHeight >= swapAreaWidth)
				swapAreaWidth = Math.min(swapAreaWidth+1, columns);
			else
				swapAreaHeight = Math.min(swapAreaHeight+1, rows);
		}	

		// get all positions of the actual swap region
		final int[] swapIndices = new int[swapAreaWidth*swapAreaHeight];
		for (int i = 0, y = 0; y < swapAreaHeight; y++)
			for (int x = 0; x < swapAreaWidth; x++)
				swapIndices[i++] = y*columns + x;
		shuffleArray(swapIndices, random);


		final int numSwapTries = (int) Math.max(1,(sampleFactor * rows * columns / swapPositions.length));
		if(doWrap) {
			for (int n = 0; n < numSwapTries; n++) {
				final int numSwapPositions = findSwapPositionsWrap(imageGrid, swapIndices, swapPositions, swapAreaWidth, swapAreaHeight); 
				doSwaps(swapPositions, numSwapPositions, imageGrid, som);
			}	
		} else {
			for (int n = 0; n < numSwapTries; n++) {
				final int numSwapPositions = findSwapPositions(imageGrid, swapIndices, swapPositions, swapAreaWidth, swapAreaHeight); 
				doSwaps(swapPositions, numSwapPositions, imageGrid, som);
			}	
		}
	}


	private int findSwapPositionsWrap(MapPlace[] imageGrid, int[] swapIndices, int[] swapPositions, int swapAreaWidth, int swapAreaHeight) {
		final int startIndex = (swapIndices.length - swapPositions.length > 0) ? random.nextInt(swapIndices.length - swapPositions.length) : 0;
		final int pos0 = random.nextInt(rows*columns);

		int numSwapPositions = 0;
		for (int j = startIndex; j < swapIndices.length && numSwapPositions < swapPositions.length; j++) {			
			int d = pos0 + swapIndices[j]; 
			int x = d % columns;
			int y = (d / columns) % rows;
			int pos = y * columns + x;

			if (imageGrid[pos] == null || imageGrid[pos].isSwapable()) 
				swapPositions[numSwapPositions++] = pos;
		}	

		return swapPositions.length;
	}


	private int findSwapPositions(MapPlace[] imageGrid, int[] swapIndices, int[] swapPositions, int swapAreaWidth, int swapAreaHeight) {

		// calculate start position of swap area
		final int pos0 = random.nextInt(rows*columns);
		final int x0 =  pos0 % columns;
		final int y0 =  pos0 / columns;

		int xStart = Math.max(0, x0 - swapAreaWidth/2);
		int yStart = Math.max(0, y0 - swapAreaHeight/2);
		if (xStart + swapAreaWidth > columns)
			xStart = columns-swapAreaWidth;
		if (yStart + swapAreaHeight > rows)
			yStart = rows-swapAreaHeight;

		final int startIndex = (swapIndices.length - swapPositions.length > 0) ? random.nextInt(swapIndices.length - swapPositions.length) : 0;
		int numSwapPositions = 0;
		for (int j = startIndex; j < swapIndices.length && numSwapPositions < swapPositions.length; j++) {
			int dx = swapIndices[j] % columns;
			int dy = swapIndices[j] / columns;

			int x = (xStart + dx) % columns;
			int y = (yStart + dy) % rows;
			int pos = y * columns + x;

			if (imageGrid[pos] == null || imageGrid[pos].isSwapable()) 
				swapPositions[numSwapPositions++] = pos;
		}

		return numSwapPositions;
	}	

	private void doSwaps(int[] swapPositions, int numSwapPositions, MapPlace[] imageGrid, float[][] som) { 

		int numValid = 0;
		for (int i = 0; i < numSwapPositions; i++) {
			final int swapPosition = swapPositions[i];
			final MapPlace swapedElement = swapedElements[i] = imageGrid[swapPosition];

			// handle holes
			if (swapedElement != null) {
				fvs[i] = swapedElement.getFloatFeature();
				numValid++;
			}
			else 
				fvs[i] = som[swapPosition]; // hole

			somFvs[i] = som[swapPosition];
		}

		if (numValid > 0) {
			final int[][] distLut = calcDistLutL2Int(fvs, somFvs, numSwapPositions);
			final int[] permutation = JonkerVolgenantSolver.computeAssignment(distLut, numSwapPositions);	

			for (int i = 0; i < numSwapPositions; i++) 
				imageGrid[swapPositions[permutation[i]]] = swapedElements[i]; 
		}
	}


	private int[][] calcDistLutL2Int(float[][] fv, float[][] mv, int size) {

		float max = 0;
		for (int i = 0; i < size; i++) 
			for (int j = 0; j < size; j++) {
				float val = distLutF[i][j] = getL2DistanceInt(fv[i], mv[j]);
				if (val > max)
					max = val;
			}

		for (int i = 0; i < size; i++) 
			for (int j = 0; j < size; j++) 
				distLut[i][j] = (int) (QUANT * distLutF[i][j] / max + 0.5);

		return distLut;
	}

	private static final int getL2DistanceInt(final float[] fv1, final float[] fv2) {

		float dist = 0;
		for (int i = 0; i < fv1.length; i++) {
			float d = fv1[i] - fv2[i];
			dist += d*d; 	
		}
		return (int) dist;
	}


	/**
	 * Representing the input data.
	 * 
	 * @author Nico Hezel
	 */
	public static class MapPlace {
		private final int id;
		private final float[] feature;
		private final boolean isSwappable;

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