/*-
 * #%L
 * This file is part of QuPath.
 * %%
 * Copyright (C) 2014 - 2016 The Queen's University of Belfast, Northern Ireland
 * Contact: IP Management (ipmanagement@qub.ac.uk)
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 * #L%
 */

package qupath.ext.sptx;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.Writer;
import java.net.URLDecoder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.gson.Gson;

import javafx.beans.property.StringProperty;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.gui.scripting.QPEx;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.measurements.MeasurementList;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.plugins.AbstractTileableDetectionPlugin;
import qupath.lib.plugins.ObjectDetector;
import qupath.lib.plugins.PluginRunner;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

/**
 * Default command for cell detection within QuPath, assuming either a nuclear or cytoplasmic staining.
 * <p>
 * To automatically classify cells as positive or negative along with detection, see {@link PositiveCellDetection}.
 * <p>
 * To quantify membranous staining see {@link WatershedCellMembraneDetection}.
 * 
 * @author Pete Bankhead
 *
 */
public class SpTxSingleCellGeneExpressionPrediction extends AbstractTileableDetectionPlugin<BufferedImage> {
	private static final boolean COMPILE_TIME = true;
	private static final String COMPILE_TIME_PYTHON_LOCATION = "/opt/anaconda3/envs/devel/bin/python";
	private static final String COMPILE_TIME_CODE_LOCATION = "/workspace/10x/qupath-0.3.1-SNAPSHOT/qupath-extension-sptx/stdnet_kernel";
	
	protected boolean parametersInitialized = false;
	final private static StringProperty m_pythonLocation = PathPrefs.createPersistentPreference("pythonLocation", null);	
		
	private final static Logger logger = LoggerFactory.getLogger(SpTxSingleCellGeneExpressionPrediction.class);
	
	private static int m_samplingFeatureSize;
	private static List<PathObject> m_availabelObjList;
	


	ParameterList params;
	
	static class STDNETPredictionParam {
		private String modelName;
		private String imageSetPath;
		private int imageCount;
	}

	static class STDNETModelArgParam {
		private String modelName;
	}
	
	static class STDNETModelArgResult {
		private int samplingSize;
		private float pixelSize;
	}
	
	static class STDNETModelList {
		private List<String> modelList;
	}
	
	static class ObjectClassification implements ObjectDetector<BufferedImage> {
		private List<PathObject> pathObjects = null;
			
		
		private Map<String, Map<String, Double>> python_sptx_stdnet_prediction(final String modelName, final int imageCount, final String imageSetPath) throws Exception {
			try {			    
				// Create a timestamp for temporary files
				final long timeStamp = System.nanoTime();

			    // Prepare parameter file for MIL4CTD
				final STDNETPredictionParam param = new STDNETPredictionParam();
				param.modelName = modelName;
				param.imageCount = imageCount;
				param.imageSetPath = imageSetPath;
				
				// Define a json file for storing parameters
				final File paramFile = File.createTempFile("stdnet_param-" + timeStamp + "-", null);
				// paramFile.deleteOnExit();
				
			    // create a writer
				final Writer paramFileWriter = new FileWriter(paramFile);

				// Obtain the temporary file name and path
				final String paramFilePath = paramFile.getAbsolutePath();

			    // convert map to JSON File
			    new Gson().toJson(param, paramFileWriter);

			    // close the writer
			    paramFileWriter.close();

				// Define a json file for storing parameters
				final File resultFile = File.createTempFile("stdnet_result-" + timeStamp + "-", null);
				// resultFile.deleteOnExit();
				// Obtain the temporary file name and path
				final String resultFilePath = resultFile.getAbsolutePath();
				
				final String pythonLocationStr =  COMPILE_TIME?
						COMPILE_TIME_PYTHON_LOCATION:
						m_pythonLocation.get();
				
				final String mainPathUtf8 = SpTxSingleCellGeneExpressionPrediction.class.getProtectionDomain().getCodeSource().getLocation().getPath();
				final String mainPathStr = URLDecoder.decode(mainPathUtf8, "UTF-8");
				final String pythonCodeLocation = COMPILE_TIME? 
						COMPILE_TIME_CODE_LOCATION:
						mainPathStr.substring(System.getProperty("os.name").startsWith("Windows")? 1: 0, mainPathStr.lastIndexOf("/"));
						
				
				final Path pythonCodePath = Paths.get(pythonCodeLocation, "sptx_stdnet.py");						
				final String pythonCodePathStr = pythonCodePath.toString();
				
				System.out.println("SPTX_STDNET: Python Path ["+pythonLocationStr+"]");
				System.out.println("SPTX_STDNET: Program Path ["+pythonCodePathStr+"]");
				System.out.println("SPTX_STDNET: Param Path ["+paramFilePath+"]");
				System.out.println("SPTX_STDNET: Result Path ["+resultFilePath+"]");
				
				final ProcessBuilder pb = new ProcessBuilder().command(
						pythonLocationStr,
						pythonCodePathStr,
						"run_prediction",
						paramFilePath,
						resultFilePath
						);						
			
	            pb.redirectErrorStream(true);
	            final Process process = pb.start();
	            final InputStream processStdOutput = process.getInputStream();
	            final Reader r = new InputStreamReader(processStdOutput);
	            final BufferedReader br = new BufferedReader(r);
	            String line;
	            while ((line = br.readLine()) != null) {
	            	System.out.println("SPTX_STDNET: "+line);
	            }
	            

	            final Reader resultJsonReader = Files.newBufferedReader(Paths.get(resultFilePath));
	            final Gson gson = new Gson();
	            final Map<String, Map<String, Double>> result = gson.fromJson(resultJsonReader, Map.class);

	            if(result == null) throw new Exception("sptx_stdnet.py returned none!"); 
	            return result;
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				throw new Exception("SpTx Cell Gene Expression Prediction requires a ROI!"); 
			}
		}
		
		@Override
		public Collection<PathObject> runDetection(final ImageData<BufferedImage> imageData, ParameterList params, ROI pathROI) throws IOException {
			if (pathROI == null)
				throw new IOException("SpTx Cell Gene Expression Prediction requires a ROI!");
			
			final ImageServer<BufferedImage> server = (ImageServer<BufferedImage>) imageData.getServer();
			final String serverPath = server.getPath();			
			final RegionRequest tileRegion = RegionRequest.createInstance(server.getPath(), 1.0, pathROI);
			
	    	pathObjects = Collections.synchronizedList(new ArrayList<PathObject>());
			
			m_availabelObjList.parallelStream().forEach( objObject -> {
				final ROI objRoi = objObject.getROI();
				final int x = (int)(0.5+objRoi.getCentroidX());
				final int y = (int)(0.5+objRoi.getCentroidY());
				
				if(tileRegion.contains(x, y, 0, 0)) {
					synchronized(pathObjects) {
						pathObjects.add(objObject);
					}
				}
			});
						
			if(pathObjects.size() > 0) {
				final AtomicBoolean success = new AtomicBoolean(false);
				

				try {
					final String modelName = (String)params.getChoiceParameterValue("modelName");
					
					// Create a temporary directory for imageset
					final Path imageSetPath = Files.createTempDirectory("sptx_stdnet_imageset-" + System.nanoTime() + "-");
					// imageSetPath.toFile().deleteOnExit();
				    
				    // Obtain the string of imageset path name
				    final String imageSetPathString = imageSetPath.toAbsolutePath().toString();
				    final int imageCount = pathObjects.size();
					
					final AtomicBoolean payloadSuccess = new AtomicBoolean(true);
					
					IntStream.range(0, pathObjects.size()).parallel().forEachOrdered(i -> { 
						final PathObject objObject = pathObjects.get(i);
						final ROI objRoi = objObject.getROI();
					    final int x0 = (int) (0.5 + objRoi.getCentroidX() - ((double)m_samplingFeatureSize / 2.0));
					    final int y0 = (int) (0.5 + objRoi.getCentroidY() - ((double)m_samplingFeatureSize / 2.0));
					    final RegionRequest objRegion = RegionRequest.createInstance(serverPath, 1.0, x0, y0, m_samplingFeatureSize, m_samplingFeatureSize);
						
						try {
							// Read image patches from server
							final BufferedImage img = (BufferedImage)server.readBufferedImage(objRegion);
							
							//  Assign a file name by sequence
					        final String imageFileName = Integer.toString(i)+".png";
					        
					        // Obtain the absolute path of the given image file name (with the predefined temporary imageset path)
					        final Path imageFilePath = Paths.get(imageSetPathString, imageFileName);
					        
					        // Make the image file
					        File imageFile = new File(imageFilePath.toString());
					        ImageIO.write(img, "png", imageFile);
						} 
						catch (IOException e) {
							payloadSuccess.set(false);
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					});
					
					if(!payloadSuccess.get()) {
		        		final String message = "SpTx Cell Level Gene Expression Prediction data preparation failed!.";
		        		logger.warn(message);							
						throw new IOException(message);
					}
										
					// final int imageSize = params.getIntParameterValue("modelFeatureSize");
					// Run python to compute MIL4CTD
					final Map<String, Map<String, Double>> result = python_sptx_stdnet_prediction(modelName, imageCount, imageSetPathString);
					
					IntStream.range(0, imageCount).parallel().forEach(i -> {
						final Map<String, Double> predList = result.get(Integer.toString(i));
						final List<String> geneIdList = new ArrayList<String>(predList.keySet());
						final PathObject objObject = pathObjects.get(i);
						final MeasurementList measList = objObject.getMeasurementList();
						
						IntStream.range(0, geneIdList.size()).parallel().forEach(j -> {
							final String geneId = geneIdList.get(j);
							double pred = predList.get(geneId);	
							
							synchronized(measList) {
								measList.addMeasurement(geneId, pred);
							}							
						});
					});
					
					
					
//					for(int i = 0; i < imageCount; i ++) {
//						final Map<String, Double> predList = result.get(Integer.toString(i));
//						final List<String> geneIdList = new ArrayList<String>(predList.keySet());
//						final PathObject objObject = pathObjects.get(i);
//						final MeasurementList measList = objObject.getMeasurementList();
//						
//						for(int j = 0; j < geneIdList.size(); j ++) {
//							final String geneId = geneIdList.get(j);
//							double pred = predList.get(geneId);
//							
//		
//							measList.addMeasurement(geneId, pred);
//						}
//						
//						measList.close();
//					}
					
					
					
					success.set(true);
			    }
				catch (Exception e) {				    	
					e.printStackTrace();
				}
			    finally {
				    System.gc();
			    }
			}
			
			return pathObjects;
		}
		
		@Override
		public String getLastResultsDescription() {
			if (pathObjects == null) return null;
			
			final int nDetections = pathObjects.size();
			
			if (nDetections == 1) return "1 nucleus classified";
			else return String.format("%d nuclei classified", nDetections);
		}
	}
	
	@Override
	protected void preprocess(final PluginRunner<BufferedImage> pluginRunner) {
		try {			    
			final ImageData<BufferedImage> imageData = pluginRunner.getImageData();
			
			// Create a timestamp for temporary files
			final long timeStamp = System.nanoTime();

			
		    // Prepare parameter file for MIL4CTD
			final STDNETModelArgParam param = new STDNETModelArgParam();
			param.modelName = (String)getParameterList(imageData).getChoiceParameterValue("modelName");
			
			// Define a json file for storing parameters
			final File paramFile = File.createTempFile("stdnet_param-" + timeStamp + "-", null);
			// paramFile.deleteOnExit();
			
		    // create a writer
			final Writer paramFileWriter = new FileWriter(paramFile);

			// Obtain the temporary file name and path
			final String paramFilePath = paramFile.getAbsolutePath();

		    // convert map to JSON File
		    new Gson().toJson(param, paramFileWriter);

		    // close the writer
		    paramFileWriter.close();
		    
		    
			// Define a json file for storing parameters
			final File resultFile = File.createTempFile("stdnet_result-" + timeStamp + "-", null);
			// resultFile.deleteOnExit();
			// Obtain the temporary file name and path
			final String resultFilePath = resultFile.getAbsolutePath();
			
			final String pythonLocationStr =  COMPILE_TIME?
					COMPILE_TIME_PYTHON_LOCATION:
					m_pythonLocation.get();
			
			final String mainPathUtf8 = SpTxSingleCellGeneExpressionPrediction.class.getProtectionDomain().getCodeSource().getLocation().getPath();
			final String mainPathStr = URLDecoder.decode(mainPathUtf8, "UTF-8");
			
			final String pythonCodeLocation = COMPILE_TIME? 
					COMPILE_TIME_CODE_LOCATION:
					mainPathStr.substring(System.getProperty("os.name").startsWith("Windows")? 1: 0, mainPathStr.lastIndexOf("/"));
			
			final Path pythonCodePath = Paths.get(pythonCodeLocation, "sptx_stdnet.py");						
			final String pythonCodePathStr = pythonCodePath.toString();
			
			System.out.println("SPTX_STDNET: Python Path ["+pythonLocationStr+"]");
			System.out.println("SPTX_STDNET: Program Path ["+pythonCodePathStr+"]");
			System.out.println("SPTX_STDNET: Action [request_model_args]");
			System.out.println("SPTX_STDNET: Param Path ["+paramFilePath+"]");
			System.out.println("SPTX_STDNET: Result Path ["+resultFilePath+"]");
			
			final ProcessBuilder pb = new ProcessBuilder().command(
					pythonLocationStr,
					pythonCodePathStr,
					"request_model_args",
					paramFilePath,
					resultFilePath
					);						
		
            pb.redirectErrorStream(true);
            final Process process = pb.start();
            final InputStream processStdOutput = process.getInputStream();
            final Reader r = new InputStreamReader(processStdOutput);
            final BufferedReader br = new BufferedReader(r);
            String line;
            while ((line = br.readLine()) != null) {
            	System.out.println("SPTX_STDNET: "+line);
            }
            
            final Reader resultJsonReader = Files.newBufferedReader(Paths.get(resultFilePath));
            final Gson gson = new Gson();
            final STDNETModelArgResult result = (STDNETModelArgResult)gson.fromJson(resultJsonReader, STDNETModelArgResult.class);
    		
    		final ImageServer<BufferedImage> server = (ImageServer<BufferedImage>) imageData.getServer();		
    		final double imagePixelSizeMicrons = server.getPixelCalibration().getAveragedPixelSizeMicrons();		
    		final double scalingFactor = result.pixelSize / imagePixelSizeMicrons;
    		m_samplingFeatureSize = (int)(0.5+(scalingFactor*result.samplingSize));
    		
    		final PathObjectHierarchy hierarchy = imageData.getHierarchy();
    		final Collection<PathObject> selectedObjects = hierarchy.getSelectionModel().getSelectedObjects();
    		final Predicate<PathObject> pred = p -> selectedObjects.contains(p.getParent());
    		m_availabelObjList = Collections.synchronizedList(QPEx.getObjects(hierarchy, pred));
    		Collections.shuffle(m_availabelObjList);  		

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}	

	@Override
	protected void postprocess(final PluginRunner<BufferedImage> pluginRunner) {
		m_availabelObjList.clear();
		System.gc();			
	}
	
	private ParameterList buildParameterList(final ImageData<BufferedImage> imageData) {
		// TODO: Use a better way to determining if pixel size is available in microns

		final AtomicBoolean success = new AtomicBoolean(false);
		
		ParameterList params = null;

		try {
			// Create a timestamp for temporary files
			final long timeStamp = System.nanoTime();
	
			// Define a json file for storing parameters
			final File resultFile = File.createTempFile("stdnet_result-" + timeStamp + "-", null);
			// resultFile.deleteOnExit();
			// Obtain the temporary file name and path
			final String resultFilePath = resultFile.getAbsolutePath();
			
			final String pythonLocationStr = COMPILE_TIME?
					COMPILE_TIME_PYTHON_LOCATION:
					m_pythonLocation.get();
			
			final String mainPathUtf8 = SpTxSingleCellGeneExpressionPrediction.class.getProtectionDomain().getCodeSource().getLocation().getPath();
			final String mainPathStr = URLDecoder.decode(mainPathUtf8, "UTF-8");
			
			final String pythonCodeLocation = COMPILE_TIME? 
					COMPILE_TIME_CODE_LOCATION:
					mainPathStr.substring(System.getProperty("os.name").startsWith("Windows")? 1: 0, mainPathStr.lastIndexOf("/"));
			
			final Path pythonCodePath = Paths.get(pythonCodeLocation, "sptx_stdnet.py");						
			final String pythonCodePathStr = pythonCodePath.toString();
			
			System.out.println("SPTX_STDNET: Python Path ["+pythonLocationStr+"]");
			System.out.println("SPTX_STDNET: Program Path ["+pythonCodePathStr+"]");
			System.out.println("SPTX_STDNET: Action [model_list]");
			System.out.println("SPTX_STDNET: Param Path []");
			System.out.println("SPTX_STDNET: Result Path ["+resultFilePath+"]");
			
			final ProcessBuilder pb = new ProcessBuilder().command(
					pythonLocationStr,
					pythonCodePathStr,
					"model_list",
					"",
					resultFilePath
					);						
		
	        pb.redirectErrorStream(true);
	        final Process process = pb.start();
	        final InputStream processStdOutput = process.getInputStream();
	        final Reader r = new InputStreamReader(processStdOutput);
	        final BufferedReader br = new BufferedReader(r);
	        String line;
	        while ((line = br.readLine()) != null) {
	        	System.out.println("SPTX_STDNET: "+line);
	        }
	        
	        final Reader resultJsonReader = Files.newBufferedReader(Paths.get(resultFilePath));
	        final Gson gson = new Gson();
	        final STDNETModelList result = (STDNETModelList)gson.fromJson(resultJsonReader, STDNETModelList.class);
	        
			params = new ParameterList();
			params
			.addTitleParameter("Setup parameters")
			.addChoiceParameter("modelName", "Model", result.modelList.get(0), new ArrayList<String>(result.modelList), 
					"Choose the model that should be used for object classification");
			
			// params
			// .addDoubleParameter("modelPixelSizeMicrons", "Pixel size for the chosen model (default: 0.5)", 0.5, IJ.micronSymbol + "m", "Pixel size for the chosen model")
			// .addIntParameter("modelFeatureSize", "Sampling size for the chosen model (default: 28)", 28, "pixel(s)", "Sampling size for the chosen model");
			
			success.set(true);
		} catch (Exception e) {
			params = null;
			
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
		    
		    System.gc();
			
		}
				
		return params;
	}
	
	@Override
	protected boolean parseArgument(ImageData<BufferedImage> imageData, String arg) {		
		return super.parseArgument(imageData, arg);
	}

	@Override
	public ParameterList getDefaultParameterList(final ImageData<BufferedImage> imageData) {
		
		if (!parametersInitialized) {
			params = buildParameterList(imageData);
		}
		
		return params;
	}

	@Override
	public String getName() {
		return "DIAnE Object Classification";
	}

	
	@Override
	public String getLastResultsDescription() {
		return "";
	}

	@Override
	public String getDescription() {
		return "Object classification based on deep learning";
	}


	@Override
	
	protected double getPreferredPixelSizeMicrons(ImageData<BufferedImage> imageData, ParameterList params) {
		final ImageServer<BufferedImage> server = (ImageServer<BufferedImage>) imageData.getServer();
		return server.getPixelCalibration().getAveragedPixelSizeMicrons();		
		
	}


	@Override
	protected ObjectDetector<BufferedImage> createDetector(ImageData<BufferedImage> imageData, ParameterList params) {
		return new ObjectClassification();
	}


	@Override
	protected int getTileOverlap(ImageData<BufferedImage> imageData, ParameterList params) {
		return 0;
	}
		
}
