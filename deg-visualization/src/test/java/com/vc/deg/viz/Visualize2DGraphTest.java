package com.vc.deg.viz;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;

public class Visualize2DGraphTest {

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		Path graphInputFile = Paths.get("c:\\Data\\Feature\\2DGraph\\L2_K4_AddK10Eps0.2High.perfect_rng_add_only.deg");
//		Path graphInputFile = Paths.get("c:\\Data\\Feature\\2DGraph\\L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd100+0_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.remove_non_rng_edges.deg");
//		Path graphOutputFile = Paths.get("c:\\Data\\Feature\\2DGraph\\L2_KCloseTo4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.deg");
		
		FeatureSpace space = new FloatL2Space(2);
		FeatureSpace.registerFeatureSpace(space);
		DynamicExplorationGraph graph = DynamicExplorationGraph.loadGraph(graphInputFile, float.class.getSimpleName());
	
		printGraph(graph);
		paintGraph(graph, replaceExtension(graphInputFile, ".png"), 1000);
	}

	
	
	/**
	 * Paint all nodes and their edges in an image file
	 * 
	 * @param graph
	 * @param pngFile
	 * @throws IOException 
	 */
	public static void paintGraph(DynamicExplorationGraph graph, Path pngFile, int imageSize) throws IOException {
		final int border = 50;
		final int nodeSize = 32;
		final int strokeSize = 6;
		
		// find the highest value per dimension
		float[] maxPerDim = new float[2];
		graph.forEachVertex((int vertexLabel, FeatureVector fv) -> {			
			for (int dim = 0; dim < 2; dim++) 
				maxPerDim[dim] = Math.max(maxPerDim[dim], fv.readFloat(dim*Float.BYTES));
		});
		
		// create the output image
		final BufferedImage bi = new BufferedImage(imageSize+border*2, imageSize+border*2, BufferedImage.TYPE_INT_ARGB);
		final Graphics2D g2d = bi.createGraphics();
		g2d.setStroke(new BasicStroke(strokeSize));

		// draw edges
		g2d.setColor(new Color(91,155,213)); // blue
		graph.forEachVertex((int vertexLabel, FeatureVector fv) -> {
			final float xStart = fv.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
			final float yStart = fv.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;

			// compute end point of the line
			graph.forEachNeighbor(vertexLabel, (int neighborLabel, float weight) -> {
				final FeatureVector fvNeighbor = graph.getFeature(neighborLabel);
				final float xEnd= fvNeighbor.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
				final float yEnd = fvNeighbor.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;				
				g2d.drawLine((int)xStart, (int)yStart, (int)xEnd, (int)yEnd);
			});
		});
		
		// draw nodes
		g2d.setColor(Color.black);
		graph.forEachVertex((int vertexLabel, FeatureVector fv) -> {
			final float x = fv.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize;
			final float y = fv.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize;
			g2d.fillOval((int)x, (int)y, nodeSize, nodeSize);			
		});
		
		// store image
		ImageIO.write(bi, "png", pngFile.toFile());
	}
	
	
	/**
	 * Print all nodes and their edges on the console
	 * 
	 * @param graph
	 */
	public static void printGraph(DynamicExplorationGraph graph) {
		graph.forEachVertex((int vertexLabel, FeatureVector feature) -> {
			System.out.print("Neighbors of vertex "+vertexLabel+": ");
			graph.forEachNeighbor(vertexLabel, (int neighborLabel, float weight) -> {
				System.out.print(neighborLabel+", ");
			});	
			System.out.println();
		});
	}
	
	
	/**
	 * Changes the extension of a file
	 * 
	 * @param file
	 * @param newExtension
	 * @return
	 */
	public static Path replaceExtension(Path file, String newExtension) {
		String fileString = file.toString();
		int end = fileString.lastIndexOf('.');
		return Paths.get(fileString.subSequence(0, end) + newExtension);
	}
	
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
