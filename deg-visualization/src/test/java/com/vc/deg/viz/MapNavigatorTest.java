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
import java.util.Collection;
import java.util.function.IntFunction;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import com.koloboke.collect.map.IntObjMap;
import com.koloboke.collect.map.hash.HashIntObjMaps;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.HierarchicalDynamicExplorationGraph;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.graph.GraphDesigner;
import com.vc.deg.viz.filter.PreparedGraphFilter;
import com.vc.deg.viz.model.GridMap;
import com.vc.deg.viz.model.WorldMap;

/**
 * Example on how to navigate and visualize a graph on a 2D grid.
 * The {@link MapDesigner} projects parts of the graph on a 2D grid
 * and the {@link MapNavigator} keeps track of the already places elements.
 * 
 * @author Nico Hezel
 */
public class MapNavigatorTest {
	
	protected static Path inputDir = Paths.get("c:\\Data\\Images\\WebImages420\\");

	public static void main(String[] args) throws IOException {
		
		final IntObjMap<ImageData> idToImageData = getImageData(inputDir);
		final int edgesPerVertex = 4;
		final int topRankSize = idToImageData.size();
		final HierarchicalDynamicExplorationGraph hGraph = buildGraph(idToImageData.values(), edgesPerVertex, topRankSize) ;
		
		// map the feature vector to a simple float array
		final IntFunction<float[]> idToFloatFeature = (int id) -> {
			final FloatFeature fv = idToImageData.get(id).getFeature();
			float[] floatArray = new float[fv.dims()];
			for (int i = 0; i < floatArray.length; i++) 
				floatArray[i] = fv.get(i);
			return floatArray;
		};
		
		// a map navigator and designer to explore the graph
		final MapNavigator mapNavigator = new MapNavigator(new WorldMap(100, 100), new MapDesigner(hGraph, idToFloatFeature));

		// empty grid to arrange the image of the graph onto
		final GridMap localMap = new GridMap(10, 10);
		
		// place the image 0 in the middle of the grid and arrange similar images from the graph around it
		final int initialId = 0;
		final PreparedGraphFilter filter = new PreparedGraphFilter(idToImageData.keySet());
		mapNavigator.jump(localMap, initialId, 0, 0, 0, filter);
		final BufferedImage image1 = toGridToImage(localMap, idToImageData, 64);
		
		// move 3 steps to the right on the grid and fill the new empty grid cells with image from the graph
		mapNavigator.move(localMap, 3, 0, 1, 0, filter);
		final BufferedImage image2 = toGridToImage(localMap, idToImageData, 64);

		// display the grid
		final JFrame frame = new JFrame();
		final JLabel label1=new JLabel();
	    label1.setIcon(new ImageIcon(image1));
	    frame.getContentPane().add(label1, BorderLayout.EAST);
	    frame.getContentPane().add(new JLabel("-----"), BorderLayout.CENTER);
		final JLabel label2=new JLabel();
	    label2.setIcon(new ImageIcon(image2));
	    frame.getContentPane().add(label2, BorderLayout.WEST);	    
	    frame.setLocationRelativeTo(null);
	    frame.pack();
	    frame.setVisible(true);
	}
	
	/**
	 * Create a BufferedImage from the image in the grid
	 * 
	 * @param grid
	 * @param idToImageData
	 * @param thumbSize
	 * @return
	 */
	protected static BufferedImage toGridToImage(GridMap grid, IntObjMap<ImageData> idToImageData, int thumbSize) {
		final int rows = grid.rows();
		final int columns = grid.columns();
		final BufferedImage result = new BufferedImage(columns * thumbSize, rows * thumbSize, BufferedImage.TYPE_INT_ARGB);
		
		final Graphics g = result.getGraphics();
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				final BufferedImage image = idToImageData.get(grid.get(c, r)).getImage();
				
				final int x = c * thumbSize;
				final int y = r * thumbSize;
				g.drawImage(image, x, y, x+thumbSize, y+thumbSize, 0, 0, image.getWidth(), image.getHeight(), null);
			}
		}
		
		return result;
	}
	
	/**
	 * Build a new hierarchical graph
	 * 
	 * @param data
	 * @param edgesPerVertex
	 * @param topRankSize
	 * @return
	 */
	protected static HierarchicalDynamicExplorationGraph buildGraph(Collection<ImageData> data, int edgesPerVertex, int topRankSize) {
		System.out.println("Build graph ...");
		final long start = System.currentTimeMillis();
		
		final int featureSize = data.iterator().next().getFeature().dims();
		final FeatureSpace space = new FloatL2Space(featureSize);
		final HierarchicalDynamicExplorationGraph hGraph = HierarchicalDynamicExplorationGraph.newGraph(space, edgesPerVertex, topRankSize) ;
		final GraphDesigner designer = hGraph.designer();

		// offer new data to the graph designer
		for (ImageData imageData : data) 
			designer.add(imageData.getId(), imageData.getFeature());
		
		// build graph
		designer.build((long step, long added, long removed, long improved, long tries, int lastAdd, int lastRemoved) -> {
			
			if(added == topRankSize)
				designer.stop();
		});
		
		System.out.println("Building graph with "+hGraph.size()+" elements and "+hGraph.levelCount()+" levels took "+(System.currentTimeMillis()-start)+"ms");
		return hGraph;
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
				final FloatFeature feature = computeMeanColorGrid(image, 3);
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
		
		Image resultingImage = image.getScaledInstance(gridSize, gridSize, Image.SCALE_DEFAULT);
	    BufferedImage outputImage = new BufferedImage(gridSize, gridSize, BufferedImage.TYPE_INT_RGB);
	    outputImage.getGraphics().drawImage(resultingImage, 0, 0, null);
	    
		final int[] rgbArray = new int[gridSize*gridSize];
		outputImage.getRGB(0, 0, gridSize, gridSize, rgbArray, 0, gridSize);
		
		final float[] featureArray = new float[rgbArray.length*3];
		for (int i = 0; i < rgbArray.length; i++) {
			final int rgb = rgbArray[i];
			featureArray[i*3+0] = (rgb << 16) & 0xFF;
			featureArray[i*3+1] = (rgb <<  8) & 0xFF;
			featureArray[i*3+2] = (rgb <<  0) & 0xFF;
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
