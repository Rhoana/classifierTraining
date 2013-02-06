addpath('../membraneDetection');
addpath('../externalPackages/NCC');
addpath('../externalPackages/randomforest-matlab/RF_Class_C');

useGPU = false;
if gpuDeviceCount > 0
    useGPU = true;
    addpath('../gpuRandomForest');
    reset(gpuDevice)
end

skript_trainClassifier_for_membraneDetection