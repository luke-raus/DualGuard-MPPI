classdef MyDubinsCar < DynSys
  properties
    wRange    % Angle bounds
    speed     % Constant speed
    dMaxMag   % Disturbance maximum magnitude
    dims      % Dimensions that are active
  end
  
  methods
    function obj = MyDubinsCar(x, wRange, speed, dMaxMag, dims)
      % obj = DubinsCar(x, wMax, speed, dMax, dims)
      %     Dubins Car class
      %
      % Dynamics:
      %    \dot{x}_1 = v * cos(x_3) + d1
      %    \dot{x}_2 = v * sin(x_3) + d2
      %    \dot{x}_3 = u
      %         u \in [-wMax, wMax]
      %         norm(d1+d2) <= dMaxMag
      %
      % Inputs:
      %   x - state: [xpos; ypos]
      %   v - constant speed
      %   thetaMin  - minimum angle
      %   thetaMax  - maximum angle
      %   dMaxMag   - disturbance maximum magnitude
      %
      % Output:
      %   obj       - a DubinsCar2D object
      
      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end
      
      if ~iscolumn(x)
        x = x';
      end
      
      if nargin < 2
        wRange = [-1 1];
      end
      
      if nargin < 3
        speed = 5;
      end
      
      if nargin < 4
        dMaxMag = 0;
      end
      
      if nargin < 5
        dims = 1:3;
      end
      
      if numel(wRange) <2
          wRange = [-wRange; wRange];
      end
      
      % Basic vehicle properties
      obj.pdim = [find(dims == 1) find(dims == 2)]; % Position dimensions
      %obj.hdim = find(dims == 3);   % Heading dimensions
      obj.nx = length(dims);
      obj.nu = 1;
      obj.nd = 3;
      
      obj.x = x;
      obj.xhist = obj.x;
      obj.speed = speed;
      obj.wRange = wRange;
      obj.dMaxMag = dMaxMag;
      obj.dims = dims;
    end
    
  end % end methods
end % end classdef
