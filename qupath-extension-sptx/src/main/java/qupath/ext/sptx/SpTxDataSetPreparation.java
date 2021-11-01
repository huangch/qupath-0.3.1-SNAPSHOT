package qupath.ext.sptx;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.File;
import org.apache.commons.lang3.ArrayUtils;
import org.controlsfx.dialog.ProgressDialog;
import org.imgscalr.Scalr;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.control.SplitPane;
import javafx.stage.Modality;
import javax.imageio.ImageIO;

import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.gui.dialogs.ProjectDialogs;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.gui.scripting.QPEx;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class SpTxDataSetPreparation implements Runnable {
	final private static Logger logger = LoggerFactory.getLogger(SpTxDataSetPreparation.class);

	final private QuPathGUI qupath;
	
	private List<ProjectImageEntry<BufferedImage>> previousImages = new ArrayList<>();
	
	// private final SpTxDialogs dianeDialogs = new SpTxDialogs();
	
	public SpTxDataSetPreparation(final QuPathGUI qupath) {
		this.qupath = qupath;
	}
	
	class ProjectTask extends Task<Void> {
		private final Collection<ProjectImageEntry<BufferedImage>> m_imagesToProcess;
		private final int m_samplingFeatureSize;
		private final double m_samplingMPP;
		private final int m_samplingStride;
		private final int m_samplingType; 
		private final int m_samplingNum; 
		private final String m_location;
		private final List<SpTxDataSetPreparationDialog.AnnotationLabel> m_labelList;
				
		private boolean quietCancel = false;
		
		ProjectTask(final Project<BufferedImage> project, final Collection<ProjectImageEntry<BufferedImage>> imagesToProcess, final List<SpTxDataSetPreparationDialog.AnnotationLabel> labelList, final String location, final int samplingFeatureSize, final double samplingMPP, final int samplingStride, final int samplingType, final int samplingNum) {
			m_imagesToProcess = imagesToProcess;
			m_samplingFeatureSize = samplingFeatureSize;
			m_samplingMPP = samplingMPP;
			m_samplingStride = samplingStride;
			m_samplingType = samplingType;
			m_samplingNum = samplingNum;
			m_location = location;
			m_labelList = labelList;
		}
		
		public void quietCancel() {
			this.quietCancel = true;
		}

		public boolean isQuietlyCancelled() {
			return quietCancel;
		}	
		
//		class SpTxConfigResult {
//			private boolean success;
//		};
//		
//		private SpTxConfigResult config() {
//			final AtomicBoolean success = new AtomicBoolean(false);
//	
//			SpTxConfigResult result = null;
//			
//			final String labelListStr = m_labelList.stream()
//	                .map(SpTxDataSetPreparationDialog.AnnotationLabel::getLabel) // This will call person.getName()
//	                .collect(Collectors.joining(","));
//			
//	
//			try {
//	
//				
//				success.set(true);
//			} catch (Exception e) {						
//				
//				e.printStackTrace();
//			} finally {
//			    System.gc();
//			}	
//			
//			return result;
//		}
	

	
		private void upload_regions(final ImageData<BufferedImage> imageData, final String annotCls, final String label) {
			final ImageServer<BufferedImage> server = (ImageServer<BufferedImage>) imageData.getServer();
			final String serverPath = server.getPath();
			
			// File f = new File(serverPath);
			// String imageId = f.getName();
			
			final double imagePixelSize = server.getPixelCalibration().getAveragedPixelSizeMicrons();
			final double scalingFactor = m_samplingMPP / imagePixelSize;
			final int samplingFeatureSize = (int)(0.5 + scalingFactor * m_samplingFeatureSize);
			final int samplingStride = (int)(0.5 + scalingFactor * m_samplingStride);
								
			final int segmentationWidth = 1+(int)((double)(server.getWidth()-samplingFeatureSize)/(double)samplingFeatureSize);
			final int segmentationHeight = 1+(int)((double)(server.getHeight()-samplingFeatureSize)/(double)samplingFeatureSize);
			
			final AtomicBoolean success = new AtomicBoolean(false);
			
			
			final List<RegionRequest> segmentationRequestList = Collections.synchronizedList(new ArrayList<RegionRequest>());
			final List<RegionRequest> availableRequestList = Collections.synchronizedList(new ArrayList<RegionRequest>());
			
			try {
						
				IntStream.range(0, segmentationHeight).parallel().forEach(y -> {
					// for(int y = 0; y < server.getHeight(); y += samplingStride) {
					IntStream.range(0, segmentationWidth).parallel().forEach(x -> {
					// for(int x = 0; x < server.getWidth(); x += samplingStride) {
						
						final int aligned_y = samplingStride*y;
						final int aligned_x = samplingStride*x;
						
						synchronized(availableRequestList) {
							availableRequestList.add(RegionRequest.createInstance(serverPath, 1.0, aligned_x, aligned_y, samplingFeatureSize, samplingFeatureSize));
						}
					});
					// }
				});
				// }
				
				final PathObjectHierarchy hierarchy = imageData.getHierarchy();
				final List<PathObject> RoiRegions = hierarchy.getFlattenedObjectList(null).stream().filter(p ->
		    		p.isAnnotation() && 
		    		p.hasROI() && 
				    p.getPathClass() != null && 
				    p.getPathClass().toString() == annotCls
			    ).collect(Collectors.toList());
			 
			    // Get all the represented classifications
				final Set<PathClass> pathClasses = new HashSet<PathClass>();
			    RoiRegions.forEach(r -> pathClasses.add(r.getPathClass()));
			    final PathClass[] pathClassArray = pathClasses.toArray(new PathClass[pathClasses.size()]);
			    final Map<PathClass, Color> pathClassColors = new HashMap<PathClass, Color>();			 
			    IntStream.range(0, pathClasses.size()).forEach(i -> pathClassColors.put(pathClassArray[i], new Color(i+1, i+1, i+1)));
			    
			    Collections.shuffle(availableRequestList);
			    
			    final AtomicBoolean payloadSuccess = new AtomicBoolean(true);
			    
				availableRequestList.parallelStream().forEach(request-> {						
				// availableRequestList.stream().forEach(request-> {
				// for(var request: availableRequestList) { 
					if(payloadSuccess.get()) {
						try {				    		
				    		final BufferedImage img = (BufferedImage)server.readBufferedImage(request);
				    				    		
				    		final int width = img.getWidth();
				    		final int height = img.getHeight();
				    		final int x = request.getX();
				    		final int y = request.getY();
		
						    // Fill the tissues with the appropriate label
				    		final BufferedImage imgMask = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
				    		final Graphics2D g2d = imgMask.createGraphics();
						    g2d.setClip(0, 0, width, height);
						    g2d.scale(1.0, 1.0);
						    g2d.translate(-x, -y);
					
						    final AtomicInteger count = new AtomicInteger(0);
						    RoiRegions.forEach(roi_region -> {
						    // for(var roi_region: RoiRegions) {
						    	final ROI roi = roi_region.getROI();
						        if (request.intersects(roi.getBoundsX(), roi.getBoundsY(), roi.getBoundsWidth(), roi.getBoundsHeight())) {
						        	final Shape shape = roi.getShape();
						        	final Color color = pathClassColors.get(roi_region.getPathClass());
					    	        g2d.setColor(color);
					    	        g2d.fill(shape);
					    	        count.incrementAndGet();			        
						        }
						    });
						    // }
						    
						    g2d.dispose();
						    
						    if (count.get() > 0) {
						        // Extract the bytes from the image
						    	final DataBufferInt buf = (DataBufferInt)imgMask.getRaster().getDataBuffer();
						    	final int[] bufData = buf.getData();
						    	
						    	final List<Integer> byteList = Arrays.asList(ArrayUtils.toObject(bufData));
						        // Check if we actually have any non-zero pixels, if necessary -
						        // we might not if the tissue bounding box intersected the region, but the tissue itself does not
						    	
						    	if(byteList.stream().filter(b -> b != 0).count() > 0) {
						    		synchronized(segmentationRequestList) {
						    			segmentationRequestList.add(request);					    			
						    		}
						        }
						    }	
						} catch (IOException e) {
							// TODO Auto-generated catch block
							payloadSuccess.set(false);
							e.printStackTrace();
						}	
		    		}
				});
				// }
				
				if(!payloadSuccess.get()) {
	        		final String message = "Region segmentation data preparation failed!.";
	        		logger.warn(message);							
					throw new IOException(message);
				}					
					
				final int samplingNum = m_samplingNum <= segmentationRequestList.size()? m_samplingNum: segmentationRequestList.size();
				final List<RegionRequest> samplingRequestList = segmentationRequestList.subList(0, samplingNum);
	
				if(samplingNum > 0) {
	
											
					payloadSuccess.set(true);
					
					IntStream.range(0, samplingRequestList.size()).parallel().forEachOrdered(i -> { 
					// for(var request: segmentationRequestList) {	
						if(payloadSuccess.get()) {
							final RegionRequest request = samplingRequestList.get(i);
							
						    // final int x = request.getX();
						    // final int y = request.getY();
						    
					        // final int segment_x = (int) ((double) x / (double) samplingFeatureSize);
					        // final int segment_y = (int) ((double) y / (double) samplingFeatureSize);	
							
					        try {
							    final BufferedImage img = (BufferedImage)server.readBufferedImage(request);
								final BufferedImage scaledImg = Scalr.resize(img, m_samplingFeatureSize);
								
								final Path locationPath = Paths.get(m_location);
								if(!Files.exists(locationPath)) new File(locationPath.toString()).mkdirs();
								
								final Path labelPath = Paths.get(m_location, label);
								if(!Files.exists(labelPath)) new File(labelPath.toString()).mkdirs();
								
								final Path imageFilePath = Paths.get(m_location, label, Integer.toString(i)+".png");
								final File imageFile = new File(imageFilePath.toString());	
								
						        ImageIO.write(scaledImg, "png", imageFile);
						        
						        
						        
					        }
					        catch (IOException e) {
								payloadSuccess.set(false);
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
						}
					});
					// }
					
					
				}
				
				success.set(true);
		    }
		    catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		    finally {
		    	availableRequestList.clear();
		    	segmentationRequestList.clear();
			
			    System.gc();
		    }
		}
			

		
		private void upload_objects(final ImageData<BufferedImage> imageData, final String annotCls, final String label) {
			final ImageServer<BufferedImage> server = (ImageServer<BufferedImage>) imageData.getServer();
			final String serverPath = server.getPath();
			// final File f = new File(serverPath);
			// final String imageId = f.getName();
			
			final double imageMPP = server.getPixelCalibration().getAveragedPixelSizeMicrons();
			final double scalingFactor = m_samplingMPP / imageMPP;
			final int samplingFeatureSize = (int)(0.5 + scalingFactor * m_samplingFeatureSize);
			

			final AtomicBoolean success = new AtomicBoolean(false);
			
			
			try {
						
				final PathObjectHierarchy hierarchy = imageData.getHierarchy();
				final List<PathObject> pathObjects = Collections.synchronizedList(new ArrayList<PathObject>());

				final List<PathObject> availableObjects = hierarchy.getFlattenedObjectList(null);
				
				availableObjects.parallelStream().forEachOrdered(p -> {
					if(p.isAnnotation() && 
					   p.hasROI() && 
					   p.getPathClass() != null &&
					   p.getPathClass().toString().equals(annotCls)
					   ) {
						final Predicate<PathObject> pred = q -> p == q.getParent();
						pathObjects.addAll(Collections.synchronizedList(QPEx.getObjects(hierarchy, pred)));
					}						
				});
									
				Collections.shuffle(pathObjects);
				
				final int samplingNum = m_samplingNum <= pathObjects.size()? m_samplingNum: pathObjects.size(); 
				final List<PathObject> samplingPathObjects = pathObjects.subList(0, samplingNum);
				
			    final AtomicBoolean payloadSuccess = new AtomicBoolean(false);
			    
				if(samplingPathObjects.size() > 0) {
				    // IntStream.range(0, samplingPathObjects.size()).parallel().forEachOrdered(i -> { 
					for(int i = 0; i < samplingPathObjects.size(); i ++) {
						final PathObject objObject = pathObjects.get(i);
						final ROI objRoi = objObject.getROI();
					    final int x0 = (int) (0.5 + objRoi.getCentroidX() - ((double)samplingFeatureSize / 2.0));
					    final int y0 = (int) (0.5 + objRoi.getCentroidY() - ((double)samplingFeatureSize / 2.0));
					    final RegionRequest objRegion = RegionRequest.createInstance(serverPath, 1.0, x0, y0, samplingFeatureSize, samplingFeatureSize);
						
						try {
							final BufferedImage img = (BufferedImage)server.readBufferedImage(objRegion);
							final BufferedImage scaledImg = Scalr.resize(img, m_samplingFeatureSize);
							
							final Path locationPath = Paths.get(m_location);
							if(!Files.exists(locationPath)) new File(locationPath.toString()).mkdirs();
							
							final Path labelPath = Paths.get(m_location, label);
							if(!Files.exists(labelPath)) new File(labelPath.toString()).mkdirs();
							
							final Path imageFilePath = Paths.get(m_location, label, Integer.toString(i)+".png");
							final File imageFile = new File(imageFilePath.toString());	
							
					        ImageIO.write(scaledImg, "png", imageFile);
						} 
						catch (IOException e) {
							payloadSuccess.set(false);
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					// });
					}
					
					
				}
				
				success.set(true);
		    }
		    catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		    finally {
			    System.gc();  
		    }
		}

		
		@Override
		public Void call() {
			
			
			long startTime = System.currentTimeMillis();
			
			int counter = 0;
							
			for (ProjectImageEntry<BufferedImage> entry : m_imagesToProcess) {
				try {
					// Stop
					if (isQuietlyCancelled() || isCancelled()) {
						logger.warn("Script cancelled with " + (m_imagesToProcess.size() - counter) + " image(s) remaining");
						break;
					}
					
					updateProgress(counter, m_imagesToProcess.size());
					counter++;
					updateMessage(entry.getImageName() + " (" + counter + "/" + m_imagesToProcess.size() + ")");
					
					// Create a new region store if we need one
					System.gc();

					// Open saved data if there is any, or else the image itself
					ImageData<BufferedImage> imageData = (ImageData<BufferedImage>)entry.readImageData();
					if (imageData == null) {
						logger.warn("Unable to open {} - will be skipped", entry.getImageName());
						continue;
					}					
					
					for(var p: m_labelList) {
					// m_labelList.stream().forEach(p -> {
						final String cls = p.getAnnotationClassObj().getAnnotationClass();
						final String lbl = p.getLabel();

						switch(m_samplingType) {
						case 0:
							upload_regions(imageData, cls, lbl);
							break;
						case 1:
							upload_objects(imageData, cls, lbl);
							break;
						}
						
					// });
					}
					
					imageData.getServer().close();
				} catch (Exception e) {
					logger.error("Error running batch script: {}", e);
				}
			}
			
			// config();
				
			updateProgress(m_imagesToProcess.size(), m_imagesToProcess.size());
			
			long endTime = System.currentTimeMillis();
			
			long timeMillis = endTime - startTime;
			String time = null;
			if (timeMillis > 1000*60)
				time = String.format("Total processing time: %.2f minutes", timeMillis/(1000.0 * 60.0));
			else if (timeMillis > 1000)
				time = String.format("Total processing time: %.2f seconds", timeMillis/(1000.0));
			else
				time = String.format("Total processing time: %d milliseconds", timeMillis);
			logger.info("Processed {} images", m_imagesToProcess.size());
			logger.info(time);
			
			return null;
		}
		
		
		@Override
		protected void done() {
			super.done();
			// tab.setRunning(false);
			// Make sure we reset the running task
			// Platform.runLater(() -> runningTask.setValue(null));
		}
		
	}
	
	
	
	
	
	
	private BooleanProperty autoClearConsole = PathPrefs.createPersistentPreference("scriptingAutoClearConsole", true);
	private ObjectProperty<Future<?>> runningTask = new SimpleObjectProperty<>();
	
	@Override
	public void run() {
    	Project<BufferedImage> project = qupath.getProject();
		if (project == null) {
			Dialogs.showNoProjectError("Script editor");
			return;
		}
		
		// Ensure that the previous images remain selected if the project still contains them
//    		FilteredList<ProjectImageEntry<?>> sourceList = new FilteredList<>(FXCollections.observableArrayList(project.getImageList()));
		
		
	    final SplitPane spane = new SplitPane();
	    spane.setPadding(new Insets(15, 12, 15, 12));
	    
		final var listSelectionView = ProjectDialogs.createImageChoicePane(qupath, project.getImageList(), previousImages, null);
	    final var annotClsSelectionView = SpTxDataSetPreparationDialog.createClassChoicePane(qupath, listSelectionView);
	    
	    spane.getItems().addAll(listSelectionView, annotClsSelectionView);
	    
		Dialog<ButtonType> dialog = new Dialog<>();
		dialog.initOwner(qupath.getStage());
		dialog.setTitle("Select project images");
		dialog.getDialogPane().getButtonTypes().addAll(ButtonType.CANCEL, ButtonType.OK);
		dialog.getDialogPane().setContent(spane);
		dialog.setResizable(true);
		dialog.getDialogPane().setPrefWidth(800);
		dialog.initModality(Modality.APPLICATION_MODAL);
		
		
//		dialog.showAndWait().ifPresent(rs -> {
//		    if (rs == ButtonType.OK) {
//		    	final String location = SpTxDataSetPreparationDialog.getLocation();
//		    	
//				if(location.strip().length() == 0) {
//					Alert alert = new Alert(AlertType.ERROR);
//					alert.setTitle("SpTxDataSetPreparation");
//					alert.setHeaderText("Error");
//					alert.setContentText("The dataset folder is empty!");
//					alert.showAndWait().ifPresent(rs2 -> {
//					    if (rs2 == ButtonType.OK) {
//					        System.out.println("Pressed OK.");
//					    }
//					});
//				};
//		        
//		    }
//		    else {
//		    	return;
//		    }
//		});
		
	
		Optional<ButtonType> result = dialog.showAndWait();
		
		if (!result.isPresent() || result.get() != ButtonType.OK)
			return;


    	final String location = SpTxDataSetPreparationDialog.getLocation();
    	
		if(location.strip().length() == 0) {
			final Alert alert = new Alert(AlertType.ERROR);
			alert.setTitle("SpTxDataSetPreparation");
			alert.setHeaderText("Error");
			alert.setContentText("The dataset folder is empty!");
			alert.showAndWait();
		};
		
		previousImages.clear();
		previousImages.addAll(ProjectDialogs.getTargetItems(listSelectionView));

		if (previousImages.isEmpty()) 
			return;

		

		final List<ProjectImageEntry<BufferedImage>> imagesToProcess = new ArrayList<>(previousImages);
		final int featureSize = SpTxDataSetPreparationDialog.getFeatureSize();
		final double samplingMPP = SpTxDataSetPreparationDialog.getSamplingMPP();
		final int samplingStride = SpTxDataSetPreparationDialog.getSamplingStride();
		final int samplingNum = SpTxDataSetPreparationDialog.getSamplingNum();
		final int samplingType = SpTxDataSetPreparationDialog.getSamplingType();
		
	    final List<SpTxDataSetPreparationDialog.AnnotationLabel> labelList = SpTxDataSetPreparationDialog.getAnnotationClassLabelList();
		
		ProjectTask worker = new ProjectTask(project, imagesToProcess, labelList, location, featureSize, samplingMPP, samplingStride, samplingType, samplingNum);
		
		ProgressDialog progress = new ProgressDialog(worker);
		progress.initOwner(qupath.getStage());
		progress.setTitle("Batch script");
		progress.getDialogPane().setHeaderText("Batch processing...");
		progress.getDialogPane().setGraphic(null);
		progress.getDialogPane().getButtonTypes().add(ButtonType.CANCEL);
		progress.getDialogPane().lookupButton(ButtonType.CANCEL).addEventFilter(ActionEvent.ACTION, e -> {
			if (Dialogs.showYesNoDialog("Cancel batch script", "Are you sure you want to stop the running script after the current image?")) {
				worker.quietCancel();
				progress.setHeaderText("Cancelling...");
				// worker.cancel(false);
				progress.getDialogPane().lookupButton(ButtonType.CANCEL).setDisable(true);
			}
			e.consume();
		});
		
		// Clear console if necessary
		if (autoClearConsole.get()) {
			// tab.getConsoleComponent().clear();
		}
		
		// Create & run task
		runningTask.set(qupath.createSingleThreadExecutor(this).submit(worker));
		progress.show();
	}
}
