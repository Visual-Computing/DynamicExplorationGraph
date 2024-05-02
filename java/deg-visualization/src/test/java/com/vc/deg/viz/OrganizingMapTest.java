package com.vc.deg.viz;

import java.awt.BorderLayout;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.IntFunction;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import com.koloboke.collect.IntCursor;
import com.koloboke.collect.map.IntObjMap;
import com.koloboke.collect.map.hash.HashIntObjMaps;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.viz.model.GridMap;
import com.vc.deg.viz.om.FLASnoMapSorterAdapter;

/**
 * Example on how to navigate and visualize a graph on a 2D grid.
 * The {@link MapDesigner} projects parts of the graph on a 2D grid
 * and the {@link MapNavigator} keeps track of the already places elements.
 * 
 * @author Nico Hezel
 */
public class OrganizingMapTest {

	protected static Path inputDir = Paths.get("c:\\Data\\ImageSet\\WebImages\\");
	protected static int fvImageSize = 32;
	protected static int thumbSize = 32;
	protected static int runs = 32;
	
	public static void main(String[] args) throws IOException {

		final IntObjMap<ImageData> idToImageData = getImageData(inputDir);
		final int featureSize = idToImageData.values().iterator().next().getFeature().dims();
		final FeatureSpace distFunc = new FloatL2Space(featureSize);
		final IntFunction<FeatureVector> idToFloatFeature = (int id) -> {
			return idToImageData.get(id).getFeature();
		};
		
		// create random grid
		final GridMap localMap = new GridMap(24, 20);
		final IntCursor idsCursor = idToImageData.keySet().cursor();	
		for (int i = 0; idsCursor.moveNext(); i++) 
			localMap.set(i, idsCursor.elem());
		final boolean[][] inUse = new boolean[localMap.rows()][localMap.columns()];
		for (int i = 0; i < 5; i++) 
			for (int j = 0; j < 5; j++) 
				inUse[i][j] = true;
				
		// sort map
		final FLASnoMapSorterAdapter adapter1 = new FLASnoMapSorterAdapter(idToFloatFeature, distFunc);
		final long start1 = System.currentTimeMillis();
		for (int i = 0; i < runs; i++) 
			adapter1.arrangeWithHoles(localMap, inUse);
		final long runtime1 = ((System.currentTimeMillis()-start1)/runs);
		
		// visualize grid
		final BufferedImage image1 = toGridToImage(localMap, idToImageData, thumbSize, false);
		final BufferedImage image2 = toGridToImage(localMap, idToImageData, thumbSize, true);

		// display the grid image
		final JFrame frame = new JFrame("runtime="+runtime1+"ms");
		final JLabel label1=new JLabel();
		label1.setIcon(new ImageIcon(image1));
		frame.getContentPane().add(label1, BorderLayout.WEST);  
		frame.getContentPane().add(new JLabel("-----"), BorderLayout.CENTER);
		final JLabel label2=new JLabel();
		label2.setIcon(new ImageIcon(image2));
		frame.getContentPane().add(label2, BorderLayout.EAST);	  
		frame.setLocationRelativeTo(null);
		frame.pack();
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	/**
	 * Create a BufferedImage from the image in the grid
	 * 
	 * @param grid
	 * @param idToImageData
	 * @param thumbSize
	 * @return
	 */
	protected static BufferedImage toGridToImage(GridMap grid, IntObjMap<ImageData> idToImageData, int thumbSize, boolean visualizeFV) {
		final int rows = grid.rows();
		final int columns = grid.columns();
		final BufferedImage result = new BufferedImage(columns * thumbSize, rows * thumbSize, BufferedImage.TYPE_INT_ARGB);

		final Graphics g = result.getGraphics();
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				final int id = grid.get(c, r);
				if(id != -1) {
					final BufferedImage imageOrig = idToImageData.get(id).getImage();				
					final int x = c * thumbSize;
					final int y = r * thumbSize;
					if(visualizeFV) { 
						
						// draw fv
						final Image fvImage = imageOrig.getScaledInstance(fvImageSize, fvImageSize, Image.SCALE_AREA_AVERAGING);
						final Image outImage = fvImage.getScaledInstance(thumbSize, thumbSize, Image.SCALE_AREA_AVERAGING);
						g.drawImage(outImage, x, y, x+thumbSize, y+thumbSize, 0, 0, thumbSize, thumbSize, null);				
					} else
						g.drawImage(imageOrig, x, y, x+thumbSize, y+thumbSize, 0, 0, imageOrig.getWidth(), imageOrig.getHeight(), null);
				}
			}
		}

		return result;
	}
	

	/**
	 * Find all JPEG images in the directory and compute their mean color to be used as a feature vector
	 * 
	 * @param inputDir
	 * @return
	 * @throws IOException
	 */
	protected static IntObjMap<ImageData> getImageData(Path inputDir) throws IOException {
		System.out.println("Read image data ...");
		final long start = System.currentTimeMillis();

		final IntObjMap<ImageData> result = HashIntObjMaps.newMutableMap();
		try(DirectoryStream<Path> stream = Files.newDirectoryStream(inputDir, "*.jpg")) {
			for(Path file : stream) {
				final int id = result.size();
				final BufferedImage image = ImageIO.read(file.toFile());
				final FloatFeature feature = computeMeanColorGrid(image, fvImageSize);
				result.put(id, new ImageData(id, image, feature));
			}
		}

		System.out.println("Read "+result.size()+" images in "+(System.currentTimeMillis()-start)+"ms");
		return result;
	}

	
	/**
	 * Compute the mean color of the image
	 * 
	 * @param image
	 * @return
	 */
	protected static FloatFeature computeMeanColorGrid(BufferedImage image, int gridSize) {

		final Image resultingImage = image.getScaledInstance(gridSize, gridSize, Image.SCALE_AREA_AVERAGING);
		final BufferedImage outputImage = new BufferedImage(gridSize, gridSize, BufferedImage.TYPE_INT_BGR);
		outputImage.getGraphics().drawImage(resultingImage, 0, 0, null);

		final int[] rgbArray = new int[gridSize*gridSize];
		outputImage.getRGB(0, 0, gridSize, gridSize, rgbArray, 0, gridSize);

		final float[] featureArray = new float[rgbArray.length*3];
		for (int i = 0; i < rgbArray.length; i++) {
			final int rgb = rgbArray[i];
			featureArray[i*3+0] = (rgb >> 16) & 0xFF; // R
			featureArray[i*3+1] = (rgb >>  8) & 0xFF; // G
			featureArray[i*3+2] = (rgb >>  0) & 0xFF; // B
		}

		return new FloatFeature(featureArray);
	}

	/**
	 * Simple POJO for the image informations
	 * 
	 * @author Nico Hezel
	 */
	protected static class ImageData {
		protected final int id;
		protected final BufferedImage image;
		protected final FloatFeature feature;

		public ImageData(int id, BufferedImage image, FloatFeature feature) {
			super();
			this.id = id;
			this.image = image;
			this.feature = feature;
		}

		public int getId() {
			return id;
		}

		public BufferedImage getImage() {
			return image;
		}

		public FloatFeature getFeature() {
			return feature;
		}
	}

	/**
	 * L2 feature space
	 * 
	 * @author Nico Hezel
	 */
	public static class FloatL2Space implements FeatureSpace {

		protected int dims;

		public FloatL2Space(int dims) {
			this.dims = dims;
		}

		@Override
		public int featureSize() {
			return dims*Float.BYTES;
		}

		@Override
		public int dims() {
			return dims;
		}

		@Override
		public Class<?> getComponentType() {
			return float.class;
		}

		@Override
		public int metric() {
			return Metric.L2.getId();
		}

		@Override
		public boolean isNative() {
			return false;
		}

		@Override
		public float computeDistance(FeatureVector f1, FeatureVector f2) {
			final int byteSize = dims * Float.BYTES;

			float result = 0;
			for (int i = 0; i < byteSize; i+=Float.BYTES) {
				float diff = f1.readFloat(i) - f2.readFloat(i);
				result += diff*diff;
			}

			return result;
		}
	}
}
