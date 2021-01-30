% GRAY - SCOTT REACTION DIFFUSION

%-------------------------------------------------------------
icase = input('Enter 1 for a new run : ');
if (icase==1)
clear all;  
F = input('F: ');kappa = input('k: ');
% Diffusion constants for Gray-Scott kinetics
D_a = 2.e-04; D_s = 1.e-04;

% dimensions
Lx = 5.50;Ly=5.50;Nx = 257;Ny = 257;   dx = Lx/(Nx-1);dy = Ly/(Ny-1);
Tmax = 200000;   Tplot=100;
%arrays
   a = zeros(Nx, Ny);s = zeros(Nx, Ny);ah = zeros(Nx, Ny);sh = zeros(Nx, Ny);
   x = zeros(Nx, Ny);y = zeros(Nx, Ny);
% initialization 
   for i = 1:Nx,
     for j = 1:Ny
      x(i,j) = i*dx; y(i,j) = j*dy;
      a(i,j) = 1.0; + 0.01*rand ;
      s(i,j) = 0.0; + 0.01*rand;
     if((i>Nx/3)&(i<2*Nx/3)&(j>Ny/3)&(j<2*Ny/3))
        a(i,j) = 0.5+ 0.005*rand; s(i,j) = 0.25+0.0025*rand;
   end
     end;
   end;
   %--------------------------------------------
   % make the ghosts 
   for j=1:Ny
       a(1,j)= a(Nx-1,j);s(1,j) = s(Nx-1,j);
       a(Nx,j)= a(2,j);s(Nx,j)=s(2,j);
   end
   for i=1:Nx
       a(i,1) = a(i,Ny-1);s(i,1)=s(i,Ny-1);
       a(i,Ny) = a(i,2);s(i,Ny) = s(i,2);
   end
   %--------------------------------------------
   pcolor(x,y,a);shading interp;drawnow;
end

% PDE parameters for Gray-Scott kinetics
dt = 4.0e-01; fa = D_a/dx^2; fs = D_s/dx^2;
fq = 0.90; %CFL 
disp ('dt = '); dt = fq/(4.0*fa)
disp('hit any key to continue');
pause;

   % loop over time
   for t = 2:Tmax
      % loop over (x,y)
      disp('Time :'); t
      for i = 2:Nx-1
            im = i-1; ip = i+1;
         for j = 2:Ny-1
            jm = j-1; jp = j+1;
            
            axy = a(i,j); axp = a(ip,j);axm = a(im,j);ayp = a(i,jp);aym = a(i,jm);
            
            sxy = s(i,j); sxp = s(ip,j);sxm = s(im,j);syp = s(i,jp);sym = s(i,jm);
 
            ah(i,j) = axy + dt*(...
                               fa * (axp+ayp+axm+aym-4*axy) ...
                     - axy * sxy^2 + F * (1.0-axy) );
            sh(i,j) = sxy + dt*(...
                               fs * (sxp+syp+sxm+sym-4*sxy)...
                     +         axy * sxy^2 - (F + kappa) * sxy );
         end;
      end;
      
        for i = 2:Nx-1
         for j = 2:Ny-1
            
            a(i,j) = ah(i,j); s(i,j) = sh(i,j);
         end;
      end;
   %--------------------------------------------
   % make the ghosts 
   for j=1:Ny
       a(1,j) = a(Nx-1,j);s(1,j) = s(Nx-1,j);
       a(Nx,j)= a(2,j)   ;s(Nx,j)= s(2,j);
   end
   for i=1:Nx
       a(i,1)  = a(i,Ny-1);s(i,1)  = s(i,Ny-1);
       a(i,Ny) = a(i,2);   s(i,Ny) = s(i,2);
   end
   %--------------------------------------------
   if(mod(t,Tplot)==0),pcolor(x,y,a);shading interp;drawnow;end;
   end;