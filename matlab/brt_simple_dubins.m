clc
clear
%% problem parameters

x_0 = [0 0 0];
speed = 4.;   % Constant velocity

% input bounds
steerMax = 6.0;

dMax = 0.2;
% Disturbance magnitudes
%dRange = {[-dMax; -dMax; 0]; [dMax; dMax; 0]};

% control trying to min or max value function?
uMode = 'max';
dMode = 'min';

filename = 'brt_dubin_environment.mat';

%% Grid

map_x_max = 5.0;
map_y_max = 5.0;
map_x_min = -map_x_max;
map_y_min = -map_y_max;

grid_res_x = 0.05;
grid_res_y = 0.05;
grid_points_x = ((map_x_max - map_x_min)/grid_res_x) + 1 + 2;
grid_points_y = ((map_y_max - map_y_min)/grid_res_y) + 1 + 2;

grid_points_theta = 91;

N = [grid_points_x; grid_points_y; grid_points_theta];   % Number of grid points per dimension

% Pad computation grid with one point outside of map to capture walls
grid_min = [ map_x_min-grid_res_x;  map_y_min-grid_res_y;  -pi];  % Lower corner of computation domain
grid_max = [ map_x_max+grid_res_x;  map_y_max+grid_res_y;   pi];  % Upper corner of computation domain

pdDims = 3;                    % State variable 3 (phi) is periodic
g = createGrid(grid_min, grid_max, N, pdDims);

%% target set
% Obstacle is cyliner, only in x-y (so ignore theta dims)

% Obstacle course

load('dubins_obstacle_data')   % loads obsX, obsY, obsR vectors

data0 = shapeCylinder(g, [3], [obsX(1); obsY(1); 0], obsR(1));
for i=2:length(obsX)
     data0 = shapeUnion(data0, shapeCylinder(g, [3], [obsX(i); obsY(i); 0], obsR(i)));
end

pad = 1.0;
% left border rect
data0 = shapeUnion(data0, shapeRectangleByCorners(g, ...
                          [map_x_min - pad;  map_y_min - pad;  -Inf], ...
                          [map_x_min;        map_y_max + pad;   Inf] ));
% right border rect
data0 = shapeUnion(data0, shapeRectangleByCorners(g, ...
                          [map_x_max;        map_y_min - pad;  -Inf], ...
                          [map_x_max + pad;  map_y_max + pad;   Inf] ));
% top border rect
data0 = shapeUnion(data0, shapeRectangleByCorners(g, ...
                          [map_x_min - pad;  map_y_max;        -Inf], ...
                          [map_x_max + pad;  map_y_max + pad;   Inf] ));
% bottom border rect
data0 = shapeUnion(data0, shapeRectangleByCorners(g, ...
                          [map_x_min - pad;  map_y_min - pad;  -Inf], ...
                          [map_x_max + pad;  map_y_min;         Inf] ));

%% time vector
t0 = 0.0;
tMax = 6.0;
dt = 0.05;
tau = t0:dt:tMax;

%% Pack problem parameters

% Define dynamic system
sys = MyDubinsCar(x_0, steerMax, speed, dMax);

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = sys;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;
schemeData.dMode = dMode;

%% Compute value function

HJIextraArgs.stopConverge = true;

% HJIextraArgs.visualize = false;
% HJIextraArgs.visualize = true; %show plot
HJIextraArgs.visualize.valueSet = 1;
HJIextraArgs.visualize.initialValueSet = 1;
HJIextraArgs.visualize.figNum = 1; %set figure number
HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update

% see a 2D slice
HJIextraArgs.visualize.plotData.plotDims = [1 1 0]; %plot x, y, phi
HJIextraArgs.visualize.plotData.projpt = [pi/2];
HJIextraArgs.visualize.viewAngle = [0,90]; % view 2D

HJIextraArgs.saveFilename = 'temp_brt';
HJIextraArgs.saveFrequency = 5;

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, 'minVOverTime', HJIextraArgs);

%xlim([-5 5])
%ylim([-5 5])

%% Export BRT for Python
export_3d_brt_with_gradient(filename, g, data, data0, pdDims, sys)







