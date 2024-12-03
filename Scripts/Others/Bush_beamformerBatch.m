%% Batch script to run source localisation analysis 
%  Daniel Bush, UCL (2019) drdanielbush@gmail.com
%
%  This script will generate one image for each participant corresponding
%  to the change in power between two time periods in the frequency band
%  specified, with associated SPM data files containing source weights etc.
%  that are needed to run subsequent analyses in source space
%
%  This uses the LCMV beamformer algorithm with a mesh (i.e. 3D volume) of
%  potential sources (i.e. sources do not have to lie on the cortical
%  surface)
%
%  Note that it is common practice to smooth the images that are created 
%  before carrying out second level analysis in SPM. In terms of smoothing, 
%  12mm is a good starting point 


%% First, provide some details of the data files to be analysed
outStub         = 'SourceRecon1';                                                                   % Sub-folder to output beamformer data files and images to
inFiles{1}      = 'C:\Users\spacememory\Desktop\MEG Analysis Tutorial\eMdpPreprocExample.mat';      % List of input pre-processed SPM MEG data files (full path!)
% inFiles{2}      = '';
% inFiles{3}      = '';


%% Then, provide some parameter settings for the analysis
baseWin         = [1 2];        % Time window for the baseline period (or leave this empty to do one time window only, seconds)
testWin         = [2 3];        % Time window for the period of interest (NOTE that baseline and test time windows must normally be the same duration, seconds)
frqBand         = [4 8];      	% Frequency band to extract integrated power values / source weights for (Hz)
res             = 10;          	% Source grid resolution in mm (5 or 10)                
startDir        = cd;           % Store starting directory


%% Initialise the SPM batch system
spm('Defaults','eeg');
spm_jobman('initcfg');


%% Then cycle through each participant...
for s   = 1: length(inFiles)
    
    % Specify all settings for the beamformer analysis:
    % 1. Head model
    matlabbatch{1}.spm.meeg.source.headmodel.D = inFiles(s);                        % This is the SPM output file which the head tracking (i.e. fiducial) data will be loaded from 
    matlabbatch{1}.spm.meeg.source.headmodel.val = 1;                               % This is the first source reconstruction performed on the data
    matlabbatch{1}.spm.meeg.source.headmodel.comment = '';
    matlabbatch{1}.spm.meeg.source.headmodel.meshing.meshes.template = 1;           % This tells SPM you wish to use a template mesh (i.e. canonical brain), rather than an individual MRI template
    matlabbatch{1}.spm.meeg.source.headmodel.meshing.meshres = 2;                   % This tells SPM you wish the resolution of the the mesh to be 'normal' (rather than 'fine' or 'coarse')
                                                                                    % We then specify the list of fiducials (i.e. for tracking the head position over time), and how they are labelled in the data file
    matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(1).fidname = 'nas';
    matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(1).specification.select = 'nas';
    matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(2).fidname = 'lpa';
    matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(2).specification.select = 'FIL_CTF_L';
    matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(3).fidname = 'rpa';
    matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(3).specification.select = 'FIL_CTF_R';
    matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.useheadshape = 0;
    matlabbatch{1}.spm.meeg.source.headmodel.forward.eeg = 'EEG BEM';               % This tells SPM how to model EEG data - it is not relevant for us
    matlabbatch{1}.spm.meeg.source.headmodel.forward.meg = 'Single Shell';          % This tells SPM to model the brain as a single shell (for the MEG signal, i.e. not a sphere, or a set of smaller spheres)
    
    % 2. Data
    outFolder   = [inFiles{s}(1:end-4) '_' outStub];
    if ~isdir(outFolder)                                                            % Create the output data folder, if it does not exist
        mkdir(outFolder)
    end
    matlabbatch{2}.spm.tools.beamforming.data.dir = {outFolder}; clear outFolder    % This tells SPM where to put the output data
    matlabbatch{2}.spm.tools.beamforming.data.D(1) = cfg_dep('M/EEG head model specification: M/EEG dataset(s) with a forward model', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','D'));
    matlabbatch{2}.spm.tools.beamforming.data.val = 1;                              % This is the first source reconstruction performed on the data (i.e. index = 1)
    matlabbatch{2}.spm.tools.beamforming.data.gradsource = 'inv';                   % This tells SPM where to get the MEG sensor data from, in order to 'invert' the signal (i.e. go from sensor to source)
    matlabbatch{2}.spm.tools.beamforming.data.space = 'MNI-aligned';                % This tells SPM what coordinate system you wish to have the output in
    matlabbatch{2}.spm.tools.beamforming.data.overwrite = 1;                        % Overwrite any previous source localisation results...
    
    % 3. Potential sources
    matlabbatch{3}.spm.tools.beamforming.sources.BF(1) = cfg_dep('Prepare data: BF.mat file', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    matlabbatch{3}.spm.tools.beamforming.sources.reduce_rank = [2 3];               % Don't worry about this! Standard (universal) values
    matlabbatch{3}.spm.tools.beamforming.sources.keep3d = 1;
    matlabbatch{3}.spm.tools.beamforming.sources.plugin.grid.resolution = res;      % Specify the mesh resolution (mm)
    matlabbatch{3}.spm.tools.beamforming.sources.plugin.grid.space = 'MNI template';% Specify the 'coordinate system' used for the cortical mesh (a template brain)
    matlabbatch{3}.spm.tools.beamforming.sources.plugin.grid.constrain = 'iskull';  % Constrain the source locaisation results to lie within the skull
    matlabbatch{3}.spm.tools.beamforming.sources.visualise = 1;                     % Show the source mapping images as you go along!
    
    % 4. Features
    matlabbatch{4}.spm.tools.beamforming.features.BF(1) = cfg_dep('Define sources: BF.mat file', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    matlabbatch{4}.spm.tools.beamforming.features.whatconditions.all = 1;           % Which conditions to include (all of them)
    matlabbatch{4}.spm.tools.beamforming.features.woi = [min([baseWin testWin]) max([baseWin testWin])]*1000;   % Which time window to include (ms)
    matlabbatch{4}.spm.tools.beamforming.features.modality = {'MEG'};               % Which recording modality to include (MEG!)
    matlabbatch{4}.spm.tools.beamforming.features.fuse = 'no';                      % Fuse data from different recording modalities (e.g. MEG and EEG)?
    matlabbatch{4}.spm.tools.beamforming.features.plugin.cov.foi = frqBand;         % Which frequency band to include
    matlabbatch{4}.spm.tools.beamforming.features.plugin.cov.taper = 'none';        % Apply any windowing before moving to frequency space, to avoid edge effects (i.e. Hanning window?) 
    matlabbatch{4}.spm.tools.beamforming.features.regularisation.manual.lambda = 1; % How to 'regularise' the data (don't worry about this - standard parameter value)
    matlabbatch{4}.spm.tools.beamforming.features.bootstrap = false;                % Do any bootstrapping?
    
    % 5. Inverse solution (standard LCMV parameter values, don't worry
    % about these!)
    matlabbatch{5}.spm.tools.beamforming.inverse.BF(1) = cfg_dep('Covariance features: BF.mat file', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    matlabbatch{5}.spm.tools.beamforming.inverse.plugin.lcmv.orient = true;
    matlabbatch{5}.spm.tools.beamforming.inverse.plugin.lcmv.keeplf = false;
    
    % 6. Output
    matlabbatch{6}.spm.tools.beamforming.output.BF(1) = cfg_dep('Inverse solution: BF.mat file', substruct('.','val', '{}',{5}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.whatconditions.all = 1;  % Which images to generate (one for each condition)
    matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.sametrials = false;      % This is a bootstrapping option - set to false
    if ~isempty(baseWin)                                                                    % Set the time windows for each image, and the contrast
        matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.woi = [baseWin ; testWin]*1000;
        matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.contrast = [-1 1];
    else
        matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.woi = testWin*1000;
        matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.contrast = 1;
    end
    matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.foi = frqBand;           % Set the frequency band for the output image    
    matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.result = 'bycondition';  % We would like a single image for each condition (i.e. not for each trial)
    matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.scale = 1;
    matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.powermethod = 'trace';
    matlabbatch{6}.spm.tools.beamforming.output.plugin.image_power.modality = 'MEG';
    
    % 7. Write
    matlabbatch{7}.spm.tools.beamforming.write.BF(1) = cfg_dep('Output: BF.mat file', substruct('.','val', '{}',{6}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    matlabbatch{7}.spm.tools.beamforming.write.plugin.nifti.normalise = 'separate';         % Normalise the output image ('separate', can also try 'none' for no scaling)
    matlabbatch{7}.spm.tools.beamforming.write.plugin.nifti.space = 'mni';
    
    % Then run the analysis for that participant!
    spm_jobman('run',matlabbatch);
    clear matlabbatch    
    
end
cd(startDir);
clear s startDir
close all
clc