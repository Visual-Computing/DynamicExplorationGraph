package com.vs.deg.ref.graph;

import java.io.BufferedInputStream;
import java.io.DataInput;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.koloboke.collect.map.IntFloatMap;
import com.koloboke.collect.map.IntIntMap;
import com.koloboke.collect.map.hash.HashIntFloatMaps;
import com.koloboke.collect.map.hash.HashIntIntMaps;
import com.vc.deg.FeatureFactory;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.io.LittleEndianDataInputStream;
import com.vc.deg.ref.feature.PrimitiveFeatureFactories;
import com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph;
import com.vc.deg.ref.graph.VertexData;

public class GraphVersionConverter {
	private static Logger log = LoggerFactory.getLogger(GraphVersionConverter.class);

	private static final Path inputDir = Paths.get("c:\\Data\\Images\\navigu\\pixabay\\graph_old\\");
	private static final Path outputDir = Paths.get("c:\\Data\\Images\\navigu\\pixabay\\graph\\");

	public static void main(String[] args) throws IOException, ClassNotFoundException, NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		try (Stream<Path> walk = Files.walk(inputDir)) {
			for (Path inputFile : walk.filter(Files::isRegularFile).filter(f -> f.toString().endsWith(".deg")).toArray(Path[]::new)) {
				final Path outputFile = outputDir.resolve(inputDir.relativize(inputFile));
				final ArrayBasedWeightedUndirectedRegularGraph graph = convertFrom0_1_2File(inputFile);
				System.out.println(inputFile +" -> "+ outputFile +" ("+ graph.getVertexCount() +")");
				Files.createDirectories(outputFile.getParent());
				graph.writeToFile(outputFile);
			}
		}
		System.out.println("finished");
	}

	public static ArrayBasedWeightedUndirectedRegularGraph convertFrom0_1_2File(Path file) throws IOException, NoSuchFieldException {
		final String filename = file.getFileName().toString();
		final int extStart = filename.lastIndexOf('.');
		final int typeStart = filename.lastIndexOf('.', extStart-1);
		final String dType = filename.substring(typeStart+1, extStart);
		return convertFrom0_1_2File(file, dType);
	}

	public static ArrayBasedWeightedUndirectedRegularGraph convertFrom0_1_2File(Path file, String featureType) throws IOException, NoSuchFieldException {
		try(BufferedInputStream bis = new BufferedInputStream(Files.newInputStream(file))) {
			final DataInput input = new LittleEndianDataInputStream(bis);

			// read meta data
			int metric = Byte.toUnsignedInt(input.readByte());
			int dims = Short.toUnsignedInt(input.readShort());
			long vertexCount = Integer.toUnsignedLong(input.readInt());
			int edgesPerVertex = Byte.toUnsignedInt(input.readByte());

			// empty graph
			if(vertexCount == 0) {
				final FeatureSpace space = FeatureSpace.findFeatureSpace(featureType, metric, dims, false);
				return new ArrayBasedWeightedUndirectedRegularGraph(edgesPerVertex, space);
			}

			// 	featureSize	=		     filesize - meta data - (edge data + label) * vertexCount   / vertexCount
			int featureSize = (int)((Files.size(file) - 8 - ((edgesPerVertex * 8 + 4) * vertexCount)) / vertexCount);

			// generic feature factory, since we will not use the features making them all to a byte array should be fine
			final FeatureFactory featureFactory = new PrimitiveFeatureFactories.ByteFeatureFactory(featureSize);

			// a feature space which does nothing but just preserves the metric, dims, featureSize and component type
			final FeatureSpace space = new FeatureSpace() {

				final Field classNameField; 
				final Class<?> clazz;
				{
					clazz = this.getClass();
					classNameField = clazz.getClass().getDeclaredField("name"); 
					classNameField.setAccessible(true);
				}

				@Override
				public int metric() {
					return metric;
				}

				@Override
				public boolean isNative() {
					return false;
				}

				@Override
				public Class<?> getComponentType() {
					try {
						classNameField.set(clazz, featureType);
					} catch (IllegalArgumentException | IllegalAccessException e) {
						throw new RuntimeException("Could not change the class name of the feature vector component type", e);
					}
					return clazz;
				}

				@Override
				public int featureSize() {
					return featureSize;
				}

				@Override
				public int dims() {
					return dims;
				}

				@Override
				public float computeDistance(FeatureVector f1, FeatureVector f2) {
					return 0;
				}
			};

			// read the vertex data
			log.debug("Read graph from file "+file.toString());
			final List<VertexData> oldVertices = new ArrayList<>((int)vertexCount); 
			for (int i = 0; i < vertexCount; i++) {

				// read the feature vector
				final FeatureVector feature = featureFactory.read(input);

				// read the edge data
				final int[] neighborIds = new int[edgesPerVertex];
				for (int e = 0; e < edgesPerVertex; e++) 
					neighborIds[e] = input.readInt();
				final float[] weights = new float[edgesPerVertex];
				for (int e = 0; e < edgesPerVertex; e++) 
					weights[e] = input.readFloat();	
				final IntFloatMap edges = HashIntFloatMaps.getDefaultFactory().withDefaultValue(Integer.MIN_VALUE).newMutableMap(edgesPerVertex);
				for (int e = 0; e < edgesPerVertex; e++)
					edges.put(neighborIds[e], weights[e]);

				// read the label
				int label = input.readInt();

				// create the vertex data
				oldVertices.add(new VertexData(label, label, feature, edges));

				if(i % 100_000 == 0)
					log.debug("Loaded "+i+" vertices");
			}
			log.debug("Loaded "+vertexCount+" vertices");

			// convert old vertex design to new one
			final List<VertexData> vertices = convertFrom0_1_2(oldVertices);
			final IntIntMap labelToId = HashIntIntMaps.getDefaultFactory().withDefaultValue(Integer.MIN_VALUE).newMutableMap(c -> {
				vertices.forEach(vertex -> {
					c.accept(vertex.getLabel(), vertex.getId());
				});
			});
			return new ArrayBasedWeightedUndirectedRegularGraph(edgesPerVertex, vertices, labelToId, space);
		}
	}

	/**
	 * Vertex lists in version 0.1.2 and below, are not in a particular order. Furthermore the id and label are the same.
	 * Starting from 0.1.3 the lists are in order of the id. The id start by 0 and end by n, where n is the number of vertices.
	 * A vertex with the id 13 is at index (13-1) in the list.
	 * 
	 * @param input
	 * @return
	 */
	protected static List<VertexData> convertFrom0_1_2(List<VertexData> input) {
		final Map<Integer, Integer> oldIdToId = new HashMap<>(input.size());
		for (int i = 0; i < input.size(); i++) 
			oldIdToId.put(input.get(i).getId(), i);

		final List<VertexData> output = new ArrayList<>();
		for (VertexData vertex : input) {
			final IntFloatMap edges = HashIntFloatMaps.getDefaultFactory().withDefaultValue(Integer.MIN_VALUE).newMutableMap(vertex.getEdges());
			output.add(new VertexData(vertex.getLabel(), oldIdToId.get(vertex.getId()), vertex.getFeature(), edges));
		}

		return output;
	}
}