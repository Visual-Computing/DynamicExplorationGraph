package com.vc.deg.mem;


import java.util.Arrays;
import java.util.Random;
import java.util.function.BiConsumer;

/*
 * Fast Linear Assignment Sorting
 * 
 * Does not work for grids with holes.
 *  
 * TODO remove 2D code
 */
public class FLAS1D {

	public static int   QUANT = 256; // /256;  quantized distance steps
		
	// hyper parameter 
	public static int   MaxSwapPositions = 9;
	public static float SampleFactor = 1.0f;	// 1 the fraction of swaps per iteration
	public static float RadiusDecay = 0.50f;
	public static float EndRadius = 1.0f; 
	public static float InitialRadiusFactor = 0.5f;
	public static int   NumFilters = 1;
		
	protected final int maxSwapPositions;
	protected final float sampleFactor;
    protected final float radiusDecay;
    protected final float radiusEnd;
    protected final float initialRadFactor;
    protected final int numFilters;
    
    protected final BiConsumer<Integer, Float> iterationListener; 
	protected final Random random;

	
    
    private int dim = -1; 
    private int rows = 0;
    private int columns = 0;
        
    private float[][] som;  
    private int[] swaps = null;
    private int swapCounter;
    
    // temporary variables
    private int[] swapPositions;
    private float[][] fvs;
    private float[][] somFvs;
    private MapPlace[] swapedElements;
    private int[][] distLut;
    private float[][] distLutF;
    
    
      
    
	public FLAS1D(Random random) {
		this(random, null);
	}
	
	public FLAS1D(Random random, BiConsumer<Integer, Float> iterationListener) {
		this.iterationListener = iterationListener;
		this.random = random;
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
		this.dim = imageGrid[0].getFloatFeature().length;
		
		final int gridSize = imageGrid.length;
		this.som = new float[gridSize][dim];
		this.swaps  = new int[gridSize];
		this.swapCounter = 0;
		
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
		for (int iteration = 0; rad > radiusEnd; iteration++) {
			copyFeatureVectorsToSom(imageGrid);			
			
			final int radius = (int) Math.max(1, Math.round(rad));  // set the radius
			final int radiusX = Math.max(1, Math.min(columns-1, radius));
			final int radiusY = Math.max(1, Math.min(rows-1, radius));
			for (int i = 0; i < numFilters; i++) 
				filterSom(radiusX, radiusY, columns, rows, som, dim, true);
//				filterSom(radius, columns, rows, som, dim);
			
			checkRandomSwaps(radius, imageGrid, som); 
			
			// inform the listener
			if(iterationListener != null)
				iterationListener.accept(iteration, rad);
			
			rad *= radiusDecay;
		}
	}
		
	private void copyFeatureVectorsToSom(MapPlace[] imageGrid) {
		for (int pos = 0; pos < imageGrid.length; pos++)  {
			final float[] fv = imageGrid[pos].getFloatFeature();
			final float[] somCell = som[pos];
			for (int i = 0; i < fv.length; i++) 
				somCell[i] = fv[i];
		}
	}

	
	protected static void filterSom(int actRadius, int columns, int rows, float[][] som, int dim) {
		int filterSize = 2*actRadius+1;
		float[][] somH = new float[rows * columns][dim];		
		filterHmirror(som, somH, rows, columns, dim, filterSize);
		som = somH;
	}
	
	protected static void filterSom(int actRadiusX, int actRadiusY, int columns, int rows, float[][] som, int dim, boolean doWrap) {

		int filterSizeX = 2*actRadiusX+1;
		int filterSizeY = 2*actRadiusY+1;

		float[][] somH = new float[rows * columns][dim];
		
		if(doWrap) {
			filterHwrap(som, somH, rows, columns, dim, filterSizeX);
			filterVwrap(somH, som, rows, columns, dim, filterSizeY);	
		}
		else {
			filterHmirror(som, somH, rows, columns, dim, filterSizeX);
			filterVmirror(somH, som, rows, columns, dim, filterSizeY);	
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
	

	private static void filterHmirror(float[][] input, float[][] output, int rows, int columns, int dims, int filterSize) {

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
	
	
	
	// -------------------------------------------------------------------------------------------------------------
	// ---------------------------------------- Swapping and Solver part--------------------------------------------
	// -------------------------------------------------------------------------------------------------------------
	
	
	private void checkRandomSwaps(int radius, MapPlace[] imageGrid, float[][] som) {
		final int swapSize = Math.max(swapPositions.length, Math.min(2*radius+1, columns));
		final int numSwapTries = (int) Math.max(1,(sampleFactor * rows * columns / swapPositions.length));
		for (int n = 0; n < numSwapTries; n++) {
			findSwapPositions(imageGrid, swapPositions, swapSize); 
			doSwaps(swapPositions, swapPositions.length, imageGrid, som);
		}
	}
	
	private void findSwapPositions(MapPlace[] imageGrid, int[] swapPositions, int swapSize) {
		
		// calculate start position of swap area
		final int size = rows*columns;
		final int start = random.nextInt(size - swapPositions.length);
		final int range = Math.min(swapSize, size - start);
		
		int numSwapPositions = 0;
		if (swapSize > swapPositions.length) {
			swapCounter++;
			do {
				final int pos = start + random.nextInt(range);	
				if (swaps[pos] != swapCounter) {
					swaps[pos] = swapCounter;
					swapPositions[numSwapPositions++] = pos;
				}	
			}
			while (numSwapPositions < swapPositions.length);
			
		} else 
			for (int x = 0; x < swapPositions.length; x++) 
				swapPositions[x] = start + x;
	}	

	private void doSwaps(int[] swapPositions, int numSwapPositions, MapPlace[] imageGrid, float[][] som) { 
		
		for (int i = 0; i < numSwapPositions; i++) {
			final int swapPosition = swapPositions[i];
			final MapPlace swapedElement = swapedElements[i] = imageGrid[swapPosition];			
			fvs[i] = swapedElement.getFloatFeature();			
			somFvs[i] = som[swapPosition];
		}
					
		final int[][] distLut = calcDistLutL2Int(fvs, somFvs, numSwapPositions);
		final int[] permutation = JonkerVolgenantSolver.computeAssignment(distLut, numSwapPositions);
		for (int i = 0; i < numSwapPositions; i++) 
			imageGrid[swapPositions[permutation[i]]] = swapedElements[i]; 
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
