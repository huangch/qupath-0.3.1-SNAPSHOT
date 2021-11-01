/*
 * I don't care
 */
package qupath.ext.sptx;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.controlsfx.control.ListSelectionView;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.ListCell;
import javafx.scene.control.SelectionMode;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TableCell;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.Label;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.DirectoryChooser;
import javafx.util.Callback;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.ProjectDialogs;
import qupath.lib.images.ImageData;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.projects.ProjectImageEntry;

/**
 *
 * @author Hasan Kara <hasan.kara@fhnw.ch>
 */
public class SpTxDataSetPreparationDialog {
	final private static Logger logger = LoggerFactory.getLogger(SpTxDataSetPreparation.class);
	
    private static final ObservableList<AnnotationClass> annotClsData
  		= FXCollections.observableArrayList();
    private static final ObservableList<AnnotationLabel> data
  	    = FXCollections.observableArrayList();

    public static void clearAnnotationClass() {
    	annotClsData.clear();
    	data.clear();
    }
    
    public static void addAnnotationClass(String annotClsStr, String labelStr) {
    	int idx = annotClsData.indexOf(annotClsStr);
    	
    	if(idx == -1) {
    		annotClsData.add(new AnnotationClass(annotClsStr));
    		idx = annotClsData.size()-1;
    	}
    	
    	data.add(new AnnotationLabel(labelStr, annotClsData.get(idx)));
    }
    
    private static TableView<AnnotationLabel> m_annotLblAsmtTable;
    private static TextField m_locationText;
    private static TextField m_regionFeatureSizeText;
    private static TextField m_regionSamplingMPPText;
    private static TextField m_regionSamplingStrideText;
    private static TextField m_regionSamplingNumText;
    private static TextField m_objectFeatureSizeText;
    private static TextField m_objectSamplingMPPText;
    private static TextField m_objectSamplingNumText;
    private static TabPane m_tabPane;
    
    public static ArrayList<AnnotationLabel> getAnnotationClassLabelList() {
    	ObservableList<AnnotationLabel> selectedRows = m_annotLblAsmtTable.getItems();    	
    	return new ArrayList<AnnotationLabel>(selectedRows);    	
    }
    
    public static String getLocation() {
    	return m_locationText.getText();
    }
    
    public static int getFeatureSize() {
    	int size = -1;
    	final int idx = m_tabPane.getSelectionModel().getSelectedIndex();
    	try {
	    	switch(idx) {
	    	case 0:
	    		size = Integer.parseInt(m_regionFeatureSizeText.getText());
	    		break;
	    	case 1:
	    		size = Integer.parseInt(m_objectFeatureSizeText.getText());
	    		break;
	    	case 2:
	    		throw new Exception("error");
	    	}
    	}
    	catch(Exception e) {
    		size = -1;
    	}
    	
    	return size;
    }
    
    public static int getSamplingNum() {
    	int size = -1;
    	final int idx = m_tabPane.getSelectionModel().getSelectedIndex();
    	try {
	    	switch(idx) {
	    	case 0:
	    		size = Integer.parseInt(m_regionSamplingNumText.getText());
	    		break;
	    	case 1:
	    		size = Integer.parseInt(m_objectSamplingNumText.getText());
	    		break;
	    	case 2:
	    		throw new Exception("error");
	    	}
    	}
    	catch(Exception e) {
    		size = -1;
    	}
    	
    	return size;
    }
    
    public static double getSamplingMPP() {
    	final int idx = m_tabPane.getSelectionModel().getSelectedIndex();
    	double MPP = -1;
    	try {
	    	switch(idx) {
	    	case 0:
	    		MPP = Double.parseDouble(m_regionSamplingMPPText.getText());
	    		break;
	    	case 1:
	    		MPP = Double.parseDouble(m_objectSamplingMPPText.getText());
	    		break;
	    	case 2:
	    		throw new Exception("error");
	    	}
    	}
    	catch(Exception e) {
    		MPP = -1;
    	}
    	
    	return MPP;
    }
    
    public static int getSamplingStride() {
    	final int idx = m_tabPane.getSelectionModel().getSelectedIndex();
    	int stride = -1;
    	try {
	    	switch(idx) {
	    	case 0:
	    		stride = Integer.parseInt(m_regionSamplingStrideText.getText());
	    		break;
	    	case 1:
	    	case 2:
	    		throw new Exception("error");
	    	}
    	}
    	catch(Exception e) {
    		stride = -1;
    	}
    	
    	return stride;
    }
    
    public static int getSamplingType() {
    	return m_tabPane.getSelectionModel().getSelectedIndex();
    }
    
    public static Pane createClassChoicePane(QuPathGUI qupath, ListSelectionView<ProjectImageEntry<BufferedImage>> listSelectionView) {
        m_annotLblAsmtTable = new TableView<>();
        
        final Label taskIdLabel = new Label("Location:");
        m_locationText = new TextField("");
        
        final Button dirChsrBtn = new Button("...");
        
        dirChsrBtn.setOnAction(new EventHandler<ActionEvent>() {
            public void handle(ActionEvent e) {
            	final DirectoryChooser locationChooser = new DirectoryChooser();
                File selectedDirectory = locationChooser.showDialog(null);
                m_locationText.setText(selectedDirectory.toString());
            }
        });
        
	  	final HBox taskIdBar = new HBox();    
	  	taskIdBar.getChildren().addAll(taskIdLabel, m_locationText, dirChsrBtn);
	  	taskIdBar.setSpacing(3);
        
	  	// Regional Pane
        final Label regionFeatureSizeLabel = new Label("Feature Size:");
        m_regionFeatureSizeText = new TextField("224");
        
        final Label regionSamplingMPPLabel = new Label("Sampling MPP:");
        m_regionSamplingMPPText = new TextField("0.5");
        
        final Label regionSamplingStrideLabel = new Label("Sampling Stride:");
        m_regionSamplingStrideText = new TextField("112");
        
        final Label regionSamplingNumLabel = new Label("Sampling Number:");
        m_regionSamplingNumText = new TextField("10000");

        final GridPane regionPane = new GridPane();    
        regionPane.setAlignment(Pos.CENTER); 
         
        regionPane.add(regionFeatureSizeLabel, 0, 0);       
        regionPane.add(m_regionFeatureSizeText, 1, 0); 
        regionPane.add(regionSamplingMPPLabel, 0, 1);       
        regionPane.add(m_regionSamplingMPPText, 1, 1); 
        regionPane.add(regionSamplingNumLabel, 0, 3);       
        regionPane.add(m_regionSamplingNumText, 1, 3); 
        regionPane.add(regionSamplingStrideLabel, 0, 2);       
        regionPane.add(m_regionSamplingStrideText, 1, 2); 
        
	  	// Object Pane
        final Label objectFeatureSizeLabel = new Label("Feature Size:");
        m_objectFeatureSizeText = new TextField("28");
        final Label objectSamplingMPPLabel = new Label("Sampling MPP:");
        m_objectSamplingMPPText = new TextField("0.5");        
        final Label objectSamplingNumLabel = new Label("Sampling Number:");
        m_objectSamplingNumText = new TextField("10000");        
    	
        final GridPane objectPane = new GridPane();    
        objectPane.setAlignment(Pos.CENTER); 
         
        objectPane.add(objectFeatureSizeLabel, 0, 0);       
        objectPane.add(m_objectFeatureSizeText, 1, 0); 
        objectPane.add(objectSamplingMPPLabel, 0, 1);       
        objectPane.add(m_objectSamplingMPPText, 1, 1); 
        objectPane.add(objectSamplingNumLabel, 0, 2);       
        objectPane.add(m_objectSamplingNumText, 1, 2); 
        
        // Tab
        
        final Tab regionTab = new Tab("Regions", regionPane);
        final Tab objectTab = new Tab("Objects"  , objectPane);
        final Tab annotatedObjectTab = new Tab("Anntated Objects" , new Text("Show all boats available"));
        
        m_tabPane = new TabPane();
        m_tabPane.getTabs().add(regionTab);
        m_tabPane.getTabs().add(objectTab);
        m_tabPane.getTabs().add(annotatedObjectTab);        
  
        /*
         * create table 
         */
        
        // final TableView<Label> annotLblAsmtTable = new TableView<>();
        
        m_annotLblAsmtTable.setEditable(true);
        m_annotLblAsmtTable.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
    	
        Callback<TableColumn<AnnotationLabel, String>, TableCell<AnnotationLabel, String>> cellFactory
                = (TableColumn<AnnotationLabel, String> param) -> new EditingCell();
        Callback<TableColumn<AnnotationLabel, AnnotationClass>, TableCell<AnnotationLabel, AnnotationClass>> comboBoxCellFactory
                = (TableColumn<AnnotationLabel, AnnotationClass> param) -> new ComboBoxEditingCell(annotClsData);

        TableColumn<AnnotationLabel, String> labelCol = new TableColumn("Label");
        labelCol.setMinWidth(100);
        labelCol.setCellValueFactory(cellData -> cellData.getValue().labelProperty());
        labelCol.setCellFactory(cellFactory);
        labelCol.setOnEditCommit(
                (TableColumn.CellEditEvent<AnnotationLabel, String> t) -> {
                    ((AnnotationLabel) t.getTableView().getItems()
                    .get(t.getTablePosition().getRow()))
                    .setLabel(t.getNewValue());

                });
        
        TableColumn<AnnotationLabel, AnnotationClass> annotClsCol = new TableColumn("Annotation Class");
        annotClsCol.setMinWidth(100);
        annotClsCol.setCellValueFactory(cellData -> cellData.getValue().annotClsObjProperty());
        annotClsCol.setCellFactory(comboBoxCellFactory);
        annotClsCol.setOnEditCommit(
                (TableColumn.CellEditEvent<AnnotationLabel, AnnotationClass> t) -> {
                    ((AnnotationLabel) t.getTableView().getItems()
                    .get(t.getTablePosition().getRow()))
                    .setAnnotationClassObj(t.getNewValue());

                });

        m_annotLblAsmtTable.setItems(data);
        m_annotLblAsmtTable.getColumns().addAll(annotClsCol, labelCol);  	
		
        
	  	final Button refButton = new Button("Refrash");
        refButton.setOnAction((ActionEvent e)
            -> {            	
            	List<ProjectImageEntry<BufferedImage>> imagesToProcess = new ArrayList<>();
            	imagesToProcess.addAll(ProjectDialogs.getTargetItems(listSelectionView));
            	final List<String> annotClsStrList = new ArrayList<String>();
            	
        		for (ProjectImageEntry<BufferedImage> entry : imagesToProcess) {
        			try {
    					// Open saved data if there is any, or else the image itself
    					ImageData<BufferedImage> imageData = (ImageData<BufferedImage>)entry.readImageData();
    					if (imageData == null) {
    						logger.warn("Unable to open {} - will be skipped", entry.getImageName());
    						continue;
    					}
    					
    					// DO thing here
    					
    					final PathObjectHierarchy hierarchy = imageData.getHierarchy();
    					
    					for(var p: hierarchy.getFlattenedObjectList(null)) {
    						if(p.isAnnotation() && 
    						   p.hasROI() && 
    						   p.getPathClass() != null && 
    						   annotClsStrList.indexOf(p.getPathClass().toString()) == -1) {
    							annotClsStrList.add(p.getPathClass().toString());
    						}
    					}
    					
    					imageData.getServer().close();
    				} catch (Exception e1) {
    					logger.error("Error running batch script: {}", e1);
    				}    					
    			}
        		
        		clearAnnotationClass();
        		
				annotClsStrList.stream().forEach(p -> {
					addAnnotationClass(p, p);	
				});
            }
        );
        
	  	final Button delButton = new Button("Delete");
        delButton.setOnAction((ActionEvent e)
            -> {
            	ObservableList<AnnotationLabel> selectedRows = m_annotLblAsmtTable.getSelectionModel().getSelectedItems();
            	ArrayList<AnnotationLabel> rows = new ArrayList<>(selectedRows);
            	rows.forEach(row -> m_annotLblAsmtTable.getItems().remove(row));            	
            }
        );
        
	  	final HBox buttonbar = new HBox();    
        buttonbar.getChildren().addAll(refButton, delButton);
        buttonbar.setSpacing(3);

	  	final VBox pane = new VBox();        
        pane.setSpacing(5);
        pane.setPadding(new Insets(10, 0, 0, 10));
        
        
        
        
        
        
        

        
        pane.getChildren().addAll(taskIdBar, m_tabPane, m_annotLblAsmtTable, buttonbar);
        pane.setVgrow(m_annotLblAsmtTable, Priority.ALWAYS); 
     




        return pane;
    }
    
    

    static class EditingCell extends TableCell<AnnotationLabel, String> {

        private TextField textField;

        private EditingCell() {
        }

        @Override
        public void startEdit() {
            if (!isEmpty()) {
                super.startEdit();
                createTextField();
                setText(null);
                setGraphic(textField);
                textField.selectAll();
            }
        }

        @Override
        public void cancelEdit() {
            super.cancelEdit();

            setText((String) getItem());
            setGraphic(null);
        }

        @Override
        public void updateItem(String item, boolean empty) {
            super.updateItem(item, empty);

            if (empty) {
                setText(item);
                setGraphic(null);
            } else {
                if (isEditing()) {
                    if (textField != null) {
                        textField.setText(getString());
//                        setGraphic(null);
                    }
                    setText(null);
                    setGraphic(textField);
                } else {
                    setText(getString());
                    setGraphic(null);
                }
            }
        }

        private void createTextField() {
            textField = new TextField(getString());
            textField.setMinWidth(this.getWidth() - this.getGraphicTextGap() * 2);
            textField.setOnAction((e) -> commitEdit(textField.getText()));
            textField.focusedProperty().addListener((ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) -> {
                if (!newValue) {
                    System.out.println("Commiting " + textField.getText());
                    commitEdit(textField.getText());
                }
            });
        }

        private String getString() {
            return getItem() == null ? "" : getItem();
        }
    }

    

    static class ComboBoxEditingCell extends TableCell<AnnotationLabel, AnnotationClass> {

        private ComboBox<AnnotationClass> comboBox;
        private ObservableList<AnnotationClass> annotClsData;
        
        public ComboBoxEditingCell(ObservableList<AnnotationClass> annotClsData) {
        	this.annotClsData = annotClsData;
        }

        @Override
        public void startEdit() {
            if (!isEmpty()) {
                super.startEdit();
                createComboBox(annotClsData);
                setText(null);
                setGraphic(comboBox);
            }
        }

        @Override
        public void cancelEdit() {
            super.cancelEdit();

            setText(getAnnotationClass().getAnnotationClass());
            setGraphic(null);
        }

        @Override
        public void updateItem(AnnotationClass item, boolean empty) {
            super.updateItem(item, empty);

            if (empty) {
                setText(null);
                setGraphic(null);
            } else {
                if (isEditing()) {
                    if (comboBox != null) {
                        comboBox.setValue(getAnnotationClass());
                    }
                    setText(getAnnotationClass().getAnnotationClass());
                    setGraphic(comboBox);
                } else {
                    setText(getAnnotationClass().getAnnotationClass());
                    setGraphic(null);
                }
            }
        }

        private void createComboBox(ObservableList<AnnotationClass> annotClsData) {
            comboBox = new ComboBox<>(annotClsData);
            comboBoxConverter(comboBox);
            comboBox.valueProperty().set(getAnnotationClass());
            comboBox.setMinWidth(this.getWidth() - this.getGraphicTextGap() * 2);
            comboBox.setOnAction((e) -> {
                System.out.println("Committed: " + comboBox.getSelectionModel().getSelectedItem());
                commitEdit(comboBox.getSelectionModel().getSelectedItem());
            });
//            comboBox.focusedProperty().addListener((ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) -> {
//                if (!newValue) {
//                    commitEdit(comboBox.getSelectionModel().getSelectedItem());
//                }
//            });
        }

        private void comboBoxConverter(ComboBox<AnnotationClass> comboBox) {
            // Define rendering of the list of values in ComboBox drop down. 
            comboBox.setCellFactory((c) -> {
                return new ListCell<AnnotationClass>() {
                    @Override
                    protected void updateItem(AnnotationClass item, boolean empty) {
                        super.updateItem(item, empty);

                        if (item == null || empty) {
                            setText(null);
                        } else {
                            setText(item.getAnnotationClass());
                        }
                    }
                };
            });
        }

        private AnnotationClass getAnnotationClass() {
            return getItem() == null ? new AnnotationClass("") : getItem();
        }
    }

    public static class AnnotationClass {

        private final SimpleStringProperty annotCls;

        public AnnotationClass(String annotCls) {
            this.annotCls = new SimpleStringProperty(annotCls);
        }

        public String getAnnotationClass() {
            return this.annotCls.get();
        }

        public StringProperty annotClsProperty() {
            return this.annotCls;
        }

        public void setAnnotationClass(String annotCls) {
            this.annotCls.set(annotCls);
        }

        @Override
        public String toString() {
            return annotCls.get();
        }

    }
    

    public static class AnnotationLabel {

        private final SimpleStringProperty label;
        private final SimpleObjectProperty<AnnotationClass> annotCls;

        public AnnotationLabel(String label, AnnotationClass annotCls) {
            this.label = new SimpleStringProperty(label);
            this.annotCls = new SimpleObjectProperty(annotCls);
        }

        public String getLabel() {
            return label.get();
        }

        public StringProperty labelProperty() {
            return this.label;
        }

        public void setLabel(String label) {
            this.label.set(label);
        }

        public AnnotationClass getAnnotationClassObj() {
            return annotCls.get();
        }

        public ObjectProperty<AnnotationClass> annotClsObjProperty() {
            return this.annotCls;
        }

        public void setAnnotationClassObj(AnnotationClass annotCls) {
            this.annotCls.set(annotCls);
        }
    }
}