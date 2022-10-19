package com.vs.deg.ref.graph;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import javax.imageio.ImageIO;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph;
import com.vc.deg.ref.graph.VertexData;

public class ArrayBasedWeightedUndirectedGraphTest {

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		Path graphInputFile = Paths.get("c:\\Data\\Feature\\2DGraph\\L2_K4_AddK10Eps0.2High.perfect_rng_add_only.deg");
//		Path graphInputFile = Paths.get("c:\\Data\\Feature\\2DGraph\\L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd100+0_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.remove_non_rng_edges.deg");
//		Path graphOutputFile = Paths.get("c:\\Data\\Feature\\2DGraph\\L2_KCloseTo4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.deg");
		
		FeatureSpace space = new FloatL2Space(2);
		FeatureSpace.registerFeatureSpace(space);
		ArrayBasedWeightedUndirectedRegularGraph graph = ArrayBasedWeightedUndirectedRegularGraph.readFromFile(graphInputFile, float.class.getSimpleName());
	
		printGraph(graph);
		paintGraph(graph, replaceExtension(graphInputFile, ".png"), 1000);
		
//		// slide 8
//		changesEdgesOfVertex(graph, 0, 3,6);
//		changesEdgesOfVertex(graph, 1, 2,10);
//		changesEdgesOfVertex(graph, 2, 1,7);
//		changesEdgesOfVertex(graph, 3, 0,7,10,13);
//		changesEdgesOfVertex(graph, 4, 5,6,9,11);
//		changesEdgesOfVertex(graph, 5, 4,12);
//		changesEdgesOfVertex(graph, 6, 0,4,8,11);
//		changesEdgesOfVertex(graph, 7, 2,3,11,12);
//		changesEdgesOfVertex(graph, 8, 6,9);
//		changesEdgesOfVertex(graph, 9, 4,8);
//		changesEdgesOfVertex(graph, 10, 1,3);
//		changesEdgesOfVertex(graph, 11, 4,6,7,13);
//		changesEdgesOfVertex(graph, 12, 5,7);
//		changesEdgesOfVertex(graph, 13, 3,11);
//		
//		// slide 9
//		changesEdgesOfVertex(graph, 0, 3,6);
//		changesEdgesOfVertex(graph, 1, 2,10);
//		changesEdgesOfVertex(graph, 2, 1,5,6,7);
//		changesEdgesOfVertex(graph, 3, 0,7,10,13);
//		changesEdgesOfVertex(graph, 4, 5,9);
//		changesEdgesOfVertex(graph, 5, 2,4,11,12);
//		changesEdgesOfVertex(graph, 6, 0,2,8,11);
//		changesEdgesOfVertex(graph, 7, 2,3,11,12);
//		changesEdgesOfVertex(graph, 8, 6,9);
//		changesEdgesOfVertex(graph, 9, 4,8);
//		changesEdgesOfVertex(graph, 10, 1,3);
//		changesEdgesOfVertex(graph, 11, 5,6,7,13);
//		changesEdgesOfVertex(graph, 12, 5,7);
//		changesEdgesOfVertex(graph, 13, 3,11);
//		
//		
//		graph.writeToFile(graphOutputFile);
	}

	/**
	 * Replaces the edges of the vertex with new neighbors
	 * 
	 * @param graph
	 * @param vertexIdx
	 * @param neighborIndices
	 */
	public static void changesEdgesOfVertex(ArrayBasedWeightedUndirectedRegularGraph graph, int vertexIdx, int ... neighborIndices) {
		VertexData data = graph.getVertexById(vertexIdx);
		Map<Integer,Float> edges = data.getEdges();
		edges.clear();
		
		for (int neigborIndex : neighborIndices) {
			VertexData neighbor = graph.getVertexById(neigborIndex);
			edges.put(neigborIndex, graph.getFeatureSpace().computeDistance(data.getFeature(), neighbor.getFeature()));
		}		
	}	
	
	/**
	 * Paint all nodes and their edges in an image file
	 * 
	 * @param graph
	 * @param pngFile
	 * @throws IOException 
	 */
	public static void paintGraph(ArrayBasedWeightedUndirectedRegularGraph graph, Path pngFile, int imageSize) throws IOException {
		final int border = 50;
		final int nodeSize = 32;
		final int strokeSize = 6;
		
		// find the highest value per dimension
		float[] maxPerDim = new float[2];
		for(VertexData node : graph.getVertices()) {
			final FeatureVector fv = node.getFeature();
			for (int dim = 0; dim < 2; dim++) 
				maxPerDim[dim] = Math.max(maxPerDim[dim], fv.readFloat(dim*Float.BYTES));
		}
		
		// create the output image
		final BufferedImage bi = new BufferedImage(imageSize+border*2, imageSize+border*2, BufferedImage.TYPE_INT_ARGB);
		final Graphics2D g2d = bi.createGraphics();
		g2d.setStroke(new BasicStroke(strokeSize));

		// draw edges
		g2d.setColor(new Color(91,155,213)); // blue
		for(VertexData node : graph.getVertices()) {
			final FeatureVector fv = node.getFeature();
			final float xStart = fv.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
			final float yStart = fv.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;

			// compute end point of the line
			for(Map.Entry<Integer, Float> edge : node.getEdges().entrySet()) {
				final FeatureVector fvNeighbor = graph.getVertexById(edge.getKey()).getFeature();
				final float xEnd= fvNeighbor.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize/2;
				final float yEnd = fvNeighbor.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize/2;				
				g2d.drawLine((int)xStart, (int)yStart, (int)xEnd, (int)yEnd);
			}
		}
		
		// draw nodes
		g2d.setColor(Color.black);
		for(VertexData node : graph.getVertices()) {
			final FeatureVector fv = node.getFeature();
			final float x = fv.readFloat(0) / maxPerDim[0] * imageSize + border - nodeSize;
			final float y = fv.readFloat(4) / maxPerDim[1] * imageSize + border - nodeSize;
			g2d.fillOval((int)x, (int)y, nodeSize, nodeSize);			
		}
		
		// store image
		ImageIO.write(bi, "png", pngFile.toFile());
	}
	
	/**
	 * Print all nodes and their edges on the console
	 * 
	 * @param graph
	 */
	public static void printGraph(ArrayBasedWeightedUndirectedRegularGraph graph) {
		for(VertexData data : graph.getVertices()) {		
			Map<Integer,Float> edges = data.getEdges();
			System.out.print("Neighbors of vertex "+data.getId()+": ");
			for (int neigborIndex : edges.keySet()) 
				System.out.print(neigborIndex+", ");
			System.out.println();
		}
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
