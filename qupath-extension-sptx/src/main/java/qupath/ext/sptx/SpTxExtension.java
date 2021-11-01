/*-
 * Copyright 2020-2021 QuPath developers,  University of Edinburgh
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.sptx;

import javafx.scene.control.Menu;
import qupath.lib.common.Version;
import qupath.lib.gui.ActionTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.tools.MenuTools;
import qupath.lib.plugins.PathPlugin;

/**
 * Install SpTx as an extension.
 * <p>
 * Currently this doesn't really do anything much, beyond including a reference 
 * in the listed extensions of QuPath and enabling some compatibility/update checks.
 * StarDist itself is only accessible via scripting.
 * In the future, the extension may also add a UI.
 * 
 * @author Pete Bankhead
 */
public class SpTxExtension implements QuPathExtension, GitHubProject {
	@SuppressWarnings("unchecked")
	@Override
	public void installExtension(QuPathGUI qupath) {
		Menu menu = qupath.getMenu("Extensions>SpTx Analysis Toolbox", true);
		
		MenuTools.addMenuItems(
				menu,
				ActionTools.createAction(new SpTxConfiguration(qupath), "Configuration")
				);
		
		MenuTools.addMenuItems(
				menu,
				qupath.createPluginAction("StarDist-based Nucleus Detection", SpTxStarDistCellNucleusDetection.class, null)
				);		
				
		MenuTools.addMenuItems(
				menu,
				ActionTools.createAction(new SpTxDataSetPreparation(qupath), "Training Data Preparation")
				);

		MenuTools.addMenuItems(
				menu,
				ActionTools.createAction(new SpTxVisiumAnnotationLoader(qupath), "Visium Annotation Loader")
				);
				
		MenuTools.addMenuItems(
				menu,
				qupath.createPluginAction("Single Cell Gene Expression Prediction", SpTxSingleCellGeneExpressionPrediction.class, null)
				);
		
	
	}

	@Override
	public String getName() {
		return "SpTx Extension";
	}

	@Override
	public String getDescription() {
		return "Run SpTx Extension.\n"
				+ "See the extension repository for citation information.";
	}
	
	@Override
	public Version getQuPathVersion() {
		return Version.parse("0.3.0-rc2");
	}

	@Override
	public GitHubRepo getRepository() {
		return GitHubRepo.create(getName(), "qupath", "qupath-extension-sptx");
	}

}
