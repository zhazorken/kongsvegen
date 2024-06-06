using Pkg
using Random
#Pkg.add("StructArrays")
#Pkg.add("CUDA")
#Pkg.add("Printf")
#Pkg.add("Statistics")
#Pkg.instantiate() # Only need to do this once when you started the repo in another machine
#Pkg.resolve()
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA: has_cuda_gpu, @allowscalar, CuArray
using Statistics: mean
using Oceanostics
using Rasters
#using StructArrays
using NetCDF

#+++ Preamble
rundir = @__DIR__ # `rundir` will be the directory of this file
#---

#+++ High level options
interpolated_IC = false
mass_flux = true
LES = true
ext_forcing = false

if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
#---

#++++ Construct grid
if LES
    params = (; Lx = 4480,
              Ly = 1600,
              Lz = 100,
              Nx = 225,
              Ny = 81,
              Nz = 50,
              ) 

else
    params = (; Lx = 3.5,
              Ly = 0.1,
              Lz = 1.5,
              Nx = 1000, #ideally 512
              Ny = 50, ##Int(Nx/2*3/5)
              Nz = 400, #ideally 750
              )
end

if arch == CPU() # If there's no CPU (e.g. if we wanna test shit on a laptop) let's use a smaller number of points!
    params = (; params..., Nx = 225, Ny = 81, Nz = 50)
end

# Creates a grid with near-constant spacing `refinement * Lz / Nz`
# near the bottom:
refinement = 6 # controls spacing near surface (higher means finer spaced)
stretching = 12  # controls rate of stretching at bottom


# "Warped" height coordinate
Nz = params.Nz
Nx = params.Nx
#(k) = (Nx + 1 - k) / Nx

# Linear near-surface generator
#ζ(k) = 1 + (h(k) - 1) / refinement

# Bottom-intensified stretching function
#Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

# Generating function
#x_faces(k) =  params.Lx * (ζ(Nx-k+1) * Σ(Nx-k+1) -1) + params.Lx
#x_faces(k) = - params.Lx * (ζ(k) * Σ(k) - 1)
#need to stretch dx from 5e-6 to .02
#grid = RectilinearGrid(arch,
#                       size = (params.Nx, params.Ny, params.Nz),
#                       x = (0, +params.Lx), #x = (0, params.Lx),
#                       y = (-params.Ly/2, +params.Ly/2),
#                       z =  (-params.Lz,0), #z_faces,
#                       topology = (Bounded, Bounded, Bounded))
#@info "Grid" grid



underlying_grid = RectilinearGrid(arch,
                       size = (params.Nx, params.Ny, params.Nz),
                       x = (0, params.Lx),
                       y = (0, +params.Ly),
                       z = (-params.Lz, 0),  #z_faces,
                       halo = (4, 4, 4),
                       topology = (Bounded, Bounded, Bounded))


#----


#bottom(x,y) =  max(- x*160/8000, -(x-8000)*40/42000-160)

#bottom(x,y)=min(0,-34+(y+600)*(9/100))

 bottom = ncread("bottom.nc", "bottom")


 #bottom(x,y)=-100

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

@info "Grid" grid  


#----

#++++ Creates a dictionary of simulation parameters
#if mass_flux # m/s
#    u_west = .03 # mass inflow through west boundary
#else
#    u₁_west = 0 # No mass flux
#end

# Not necessary, but makes organizing simulations easier and facilitates running on GPUs
params = (; params...,
          N²₀ = 0e-6, # 1/s (stratification frequency)
          #z₀ = 300, # m
          #σz_west = 10, # m (height of the dense water inflow at the west boundary)
          u₁_west = 62/(200*4), #u₁_west, discharge/(DY*DZ)          
          ℓ₀ = 0.1, # m (roughness length)
          σ = 5.0seconds, # s (relaxation rate for sponge layer)
          #uₑᵥₐᵣ = 0.00, # m/s (velocity variation along the z direction of the east boundary)
          u_b = 0.00, #external nudged u
          v_b = 0.00, #external nudged v
          )


#----

#++++ Conditions opposite to the ice wall (@ infinity)
if LES
  #  b∞(z, parameters) = parameters.N²₀ * z # Linear stratification in the interior (far from ice face)

T∞(z, parameters)=-0.25*4*tanh((z-0.)/4.)+2.6;
S∞(z, parameters)=-2.7*4*tanh((z-0.)/4.)+22;
d∞(z, parameters)=-0.5*tanh(z+20) + 0.5; 

   # T∞(z, parameters)=-0.9 *tanh((z - 120.) / (10. / 2.))+7+max(.1/100. *(100. -z),0)
   # S∞(z, parameters)=-2. *tanh((z - 120.) / (15. / 2.))+26+max(.1/100. *(100. -z),0)

   # T∞(x, parameters) = -0.375 - 0.05 * x
   # S∞(x, parameters) = 34.42 - 0.0223 * x
    u∞(x, parameters) = @allowscalar u_b
    v∞(x, parameters) = @allowscalar v_b
end
#----

#++++ Western BCs
if LES
    b_west(y, z, t, p) = b∞(z, p) #+ p.b₁_west / (1 + exp((z-p.z₀)/p.σz_west))

    if mass_flux
        u_west(y, z, t, p) = p.u₁_west # / (1 + exp((z-p.z₀)/p.σz_west))
    end
end

#++++ Drag BC for v and w
if LES
    const κ = 0.4 # von Karman constant
    x₁ₘₒ = @allowscalar xnodes(grid, Center())[1] # Closest grid center to the bottom
    cᴰ = (κ / log(x₁ₘₒ/params.ℓ₀))^2 # Drag coefficient
    cᴰ = 2.5e-3
    @inline drag_w(x, y, t, v, w, p) = - p.cᴰ * √(w^2 + v^2) * w
    @inline drag_v(x, y, t, v, w, p) = - p.cᴰ * √(w^2 + v^2) * v

    drag_bc_w = FluxBoundaryCondition(drag_w, field_dependencies=(:v, :w), parameters=(cᴰ=cᴰ,))
    drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:v, :w), parameters=(cᴰ=cᴰ,))
end
#----
#----

#++++ Eastern BCs
if mass_flux # What comes in has to go out
    params = (; params..., u_out = mean(u_west.(0, grid.zᵃᵃᶜ[1:Nz], 0, (params,))))
    # The function below allows a net mass flux out of exactly u_out, but with variations in the form of
    # a sine function. After a mean in z, this function return exactly u_out.
    u∞(y, z, t, p) = p.u_out #+ p.uₑᵥₐᵣ*sin(-2π*z/grid.Lz) 
else
    params = (; params..., u_out = 0)
end
#----

#++++ Eastern sponge layer 
# (smoothes out the mass flux and gets rid of some of the build up of buoyancy)

const Lz = params.Lz
const Lx = params.Lx

#function calculate_seed_value(y) #need to be separate function to call on cpus
#    t = time()
#    seed_value = Int(floor(t + y))
#    return seed_value
#end

  #  t=time()  
  #  seed_value = Int(floor(t+y))
@inline function west_mask(x, y, z, t, params)
#seed_value = calculate_seed_value(y)
  #  t=time()
  #  seed_value = Int(floor(t+y))
 #   Random.seed!(seed_value)
    z₀ = -27  #inner location
    z₁ = -23  #outer location
    x₀ = 0  #inner location
    x₁ = 20  #outer location
    y₀ = 700  #inner location
    y₁ = 900  #outer location
    if (z₀ <= z <= z₁) && (x₀ <= x <= x₁) && (y₀ <= y <= y₁)
        return (x₁-x)/x₁*(-0.1*cos(8*pi*5/(y₁-y₀)*y)+1+1*(rand()-0.5))   #5 periods = 10 peaks
    else
        return 0.0
    end
end


@inline function east_mask(x, y, z, params)
    z₀ = params.Lz-20  #inner location
    z₁ = params.Lz  #outer location
    x₀ = params.Lx-200  #inner location
    x₁ = params.Lx  #outer location
    if (z₀ <= z <= z₁) && (x₀ <= x <= x₁) 
            return (x-x₀)/(x₁-x₀)
    else
        return 0.0
    end
end

@inline function top_bot_mask_cos(x, y, z)
    x₀ = 0.1  #inner location
    x₁ = 0.0  #outer location
    
    x2 = Lx-0.1  #inner location
    x3 = Lx  #outer location

    if x₀ >= x >= x₁
        return 1/2 * (1 - cos( π*(x-x₀)/(x₁-x₀) ))
    elseif x2 <= x <= x3
        return 1/2 * (1 - cos( π*(x-x2)/(x3-x2) )) 
    else
        return 0.0
    end
end


if mass_flux
    @inline sponge_u(x, y, z, t, u, p) = -west_mask(x, y, z, t, p) / p.σ * (u - p.u₁_west)-east_mask(x, y, z, p) / p.σ/10 * (u - p.u₁_west*200*4/1620/20) # Nudges east to DYout*DZout/DYDZeast 
  #  @inline sponge_v(x, y, z, t, v, p) = -west_mask_cos(x, y, z) * p.σ * v # nudges v to zero
  #  @inline sponge_w(x, y, z, t, w, p) = -west_mask_cos(x, y, z) * p.σ * w # nudges w to zero
end
#@inline sponge_u(x, y, z, t, u, p) = -min(west_mask_cos(x, y, z)+top_bot_mask_cos(x,y,z),1.0) * p.σ * (u - p.u_b) # nudges u to u∞
#@inline sponge_v(x, y, z, t, v, p) = -west_mask_cos(x, y, z) * p.σ * (v - p.v_b) # nudges v to v∞
@inline sponge_T(x, y, z, t, T, p) = -west_mask(x, y, z, t, p) / p.σ * (T - 0)-east_mask(x, y, z, p) / p.σ/30 * (T - T∞(z, p)) # nudges T to T∞
@inline sponge_S(x, y, z, t, S, p) = -west_mask(x, y, z, t, p) / p.σ * (S - 0)-east_mask(x, y, z, p) / p.σ/30 * (S - S∞(z, p)) # nudges S to S∞
@inline sponge_c(x, y, z, t, c, p) = -west_mask(x, y, z, t, p) / p.σ * (c - 1)-east_mask(x, y, z, p) / p.σ/30 * (c - 0) # nudges S to S∞
@inline sponge_d(x, y, z, t, d, p) = -west_mask(x, y, z, t, p) / p.σ * (d - 0)-east_mask(x, y, z, p) / p.σ/30 * (d - d∞(z, p)) # nudges S to S∞
#----

#++++ Assembling forcings and BCs
if ext_forcing
  Fᵤ = Forcing(sponge_u, field_dependencies = :u, parameters = params)
  Fᵥ = Forcing(sponge_v, field_dependencies = :v, parameters = params)
  forcing = (u=Fᵤ, v=Fᵥ, T=FT, S=FS)
else
  FT = Forcing(sponge_T, field_dependencies = :T, parameters = params)
  FS = Forcing(sponge_S, field_dependencies = :S, parameters = params)
  Fc = Forcing(sponge_c, field_dependencies = :c, parameters = params)
  Fd = Forcing(sponge_d, field_dependencies = :d, parameters = params)
  forcing = (T=FT, S=FS, c=Fc)
end

if mass_flux
    Fᵤ = Forcing(sponge_u, field_dependencies = :u, parameters = params)
  #  Fᵥ = Forcing(sponge_v, field_dependencies = :v, parameters = params)
  #  Fw = Forcing(sponge_w, field_dependencies = :w, parameters = params)
    forcing = (u=Fᵤ, T=FT, S=FS, c=Fc, d=Fd)
end

#solve for dTdx and dSdx
const a_s = -5.73e-2
const L = 3.35e5
const c_w = 3974
const kappa_S=7.2e-10
const kappa_T=1.3e-7
#const dz = @allowscalar grid.Δzᵃᵃᶜ[Nz]# Lx/Nx; #the dz at the topmost gridpoint

#%function get_T0(x, y, t, T, S)
#    bb = -T- L*kappa_S/(kappa_T*c_w)  #need to define T1
#    cc = L*kappa_S*a_s*S / (kappa_T*c_w)  #need to define S1
#    return (-bb-sqrt(bb^2-4*cc))/2
#end

#function get_S0(x, y, t, T, S)
#    T0 = get_T0(x, y, t, T, S)
#    S0 = T0/a_s
#    return S0
#end

function get_T0(y,z,T)
if (-50 < y && y < 50) && (0 < z && z < 15)
    return 0
else
    return T
end
end

function get_S0(y,z,S)
if (-50 < y && y < 50) && (0 < z && z < 15)
    return 0
   else
    return S 
end
end

function get_u0(y,z)
if (-50 < y && y < 50) && (10 < z && z < 15)
    return 100/(100*15)
else
    return 0
end
end


T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0), #ValueBoundaryCondition(get_T0, field_dependencies=(:T, :S)), 
                               # west = ValueBoundaryCondition(get_T0, field_dependencies=(:T)),
                                east = FluxBoundaryCondition(0), # Hidden behind sponge layer
                                )

                                
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0), #ValueBoundaryCondition(get_S0, field_dependencies=(:T, :S)),
                              #  west = ValueBoundaryCondition(get_S0, field_dependencies=(:S)),
                                east = FluxBoundaryCondition(0), # Hidden behind sponge layer
                                )

                                
                                
u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0),
                              #  west = GradientBoundaryCondition(get_u0),
                                top = ValueBoundaryCondition(0),
                                )
w_bcs = FieldBoundaryConditions(east = ValueBoundaryCondition(0),
                                west = drag_bc_w,
                                #west = ValueBoundaryCondition(0),
                                )
v_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0), 
                                west = drag_bc_v,
                                bottom = ValueBoundaryCondition(0), 
                                )
boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs,)
#----

#++++ Lagrangian particles

n_particles = 20000;

x₀ = 50*rand(n_particles); #100*ones(n_particles); #rand(n_particles);

y₀ = (rand(n_particles).-0.5)*params.Ly;

z₀ = params.Lz*rand(n_particles); #-0.5 * ones(n_particles);

if arch == CPU()

lagrangian_particles = LagrangianParticles(x=x₀, y=y₀, z=z₀)

#=
T=3*ones(n_particles); #some random initialization. It shouldn't matter

struct CustomParticle
    x::Float64  # x-coordinate
    y::Float64  # y-coordinate
    z::Float64  # z-coordinate
    T::Float64  # Temperature
   # S::Float64  # Salinity
   # temperature_history::Vector{Float64}  # Temperature history
end



#particles = StructArray{CustomParticle}((x₀, y₀, z₀, (T=get_T(T),field_dependencies=(:T))));

particles = StructArray{CustomParticle}((x₀, y₀, z₀, T));


# Define tracked fields as a NamedTuple
tracked_fields = (T=particles.T,)

# Initialize LagrangianParticles with the StructArray and tracked fields
lagrangian_particles = LagrangianParticles(particles; tracked_fields=tracked_fields)
=#

else 

# Create LagrangianParticles with CuArrays
lagrangian_particles = LagrangianParticles(x=CuArray(x₀), y=CuArray(y₀), z=CuArray(z₀))

#particles = StructVector{Particle}(CuArray{Float64}, CuArray{Float64}, CuArray{Float64})  # Define particles with CuArray

#lagrangian_particles = LagrangianParticles(x=x₀, y=y₀, z=z₀)

end

#++++ Construct model
if LES
    closure = AnisotropicMinimumDissipation()
else
    closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(),ν=1.8e-6, κ=(T=1.3e-7, S=7.2e-10))
end

θ = 0 #90 # degrees relative to pos. x-axis

model = NonhydrostaticModel(grid = grid, 
                          #  particles=lagrangian_particles,
                            advection = WENO(grid=grid, order=5),
                            timestepper = :QuasiAdamsBashforth2, 
                            tracers = (:T, :S, :c, :d),
                            buoyancy = Buoyancy(model=SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 3.87e-5,
                            haline_contraction = 7.86e-4)), gravity_unit_vector=(-sind(θ),0,-cosd(θ))),
                            #buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 3.87e-5,
                            #haline_contraction = 7.86e-4)),
                            coriolis = FPlane(0e-4),
                            closure = closure,
                            forcing = forcing,
                            boundary_conditions = boundary_conditions,
                            )
@info "Model" model
#----

#++++ Create simulation
using Oceanostics.ProgressMessengers: BasicTimeMessenger

Δt₀ = 1/2 * minimum_zspacing(grid) / (3  + 1e-1)
simulation = Simulation(model, Δt=Δt₀,
                        stop_time = 4days, # when to stop the simulation
)

#++++ Adapt time step
wizard = TimeStepWizard(cfl=0.5, # How to adjust the time step
                        diffusive_cfl=5,
                        max_change=1.02, min_change=0.2, max_Δt=0.5/√params.N²₀, min_Δt=0.00001seconds)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2)) # When to adjust the time step
#----

#++++ Printing to screen
progress = BasicTimeMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10)) # when to print on screen

#----

@info "Simulation" simulation
#----


##Note: HERE I TRIED TO WRITE A FUNCTION to access the T[x,y,z] given the particle location x,y,z. But I couldn't figure it out

#function update_particle_temperatures!(model::NonhydrostaticModel, particles::Vector{CustomParticle})
#    function update_particle_temperatures!(model, particles)
#    for particle in particles
###        # Get the temperature at the particle's position in the model
#        x_idx = round(Int, (particle.x - grid.xᶜᵃᵃ[1]) / (grid.xᶜᵃᵃ[end] - grid.xᶜᵃᵃ[1]) * (params.Nx - 1)) + 1
#        y_idx = round(Int, (particle.y - grid.yᵃᶜᵃ[1]) / (grid.yᵃᶜᵃ[end] - grid.yᵃᶜᵃ[1]) * (params.Ny - 1)) + 1
#        z_idx = round(Int, (particle.z - grid.zᵃᵃᶜ[1]) / (grid.zᵃᵃᶜ[end] - grid.zᵃᵃᶜ[1]) * (params.Nz - 1)) + 1
#
#        # Update the particle's position indices
#        particle.x = x_idx
#        particle.y = y_idx
#        particle.z = z_idx
#        particle.T = T[x_idx, y_idx, z_idx]
#    end
#end


    #=
#function update_particle_temperatures!(particles::Vector{Union{CustomParticle, Int64}})
    for particle in particles
        if particle isa CustomParticle
            # Get the temperature at the particle's position in the model
            x_idx = round(Int, (particle.x - grid.xᶜᵃᵃ[1]) / (grid.xᶜᵃᵃ[end] - grid.xᶜᵃᵃ[1]) * (params.Nx - 1)) + 1
            y_idx = round(Int, (particle.y - grid.yᵃᶜᵃ[1]) / (grid.yᵃᶜᵃ[end] - grid.yᵃᶜᵃ[1]) * (params.Ny - 1)) + 1
            z_idx = round(Int, (particle.z - grid.zᵃᵃᶜ[1]) / (grid.zᵃᵃᶜ[end] - grid.zᵃᵃᶜ[1]) * (params.Nz - 1)) + 1

            # Update the particle's position indices
            particle.x = x_idx
            particle.y = y_idx
            particle.z = z_idx
            particle.T = T[x_idx, y_idx, z_idx]
        end
    end
end


# Function to update particle temperatures
function update_particle_temperatures!(model)
    particles = model.particles.particles
    for i in 1:length(particles.T)
        particles.T[i] += 1.0  # Example operation: incrementing temperature by 1.0
    end
end

# Update particle temperatures
update_particle_temperatures!(model)
=#

#++++ Impose initial conditions
u, v, w =  model.velocities

T = model.tracers.T
S = model.tracers.S
c = model.tracers.c
d = model.tracers.d

if interpolated_IC
    filename = "IC_part.nc"
    @info "Imposing initial conditions from existing NetCDF file $filename"

    using Rasters
    rs = RasterStack(filename, name=(:u, :v, :w, :T, :S))

    u[1:grid.Nx+1, 1:grid.Ny, 1:grid.Nz] .= Array(rs.u[ Ti=Near(Inf) ])
    v[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= Array(rs.v[ Ti=Near(Inf) ])
    w[1:grid.Nx, 1:grid.Ny, 1:grid.Nz+1] .= Array(rs.w[ Ti=Near(Inf) ])

    S[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= Array(rs.S[ Ti=Near(Inf) ])
    T[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= Array(rs.T[ Ti=Near(Inf) ])

else
    @info "Imposing initial conditions from scratch"

    const T_i=-0.3
    const S_i=T_i/a_s
    const Le=kappa_T/kappa_S
    const delta_B=0.0017

    T_ic(x, y, z) = T∞(z, params)
 #   T_ic(x,y,z) =  2/pi*(T∞(x,params)-T_i)*atan((Lz-z)/delta_B)+T_i

    S_ic(x, y, z) = S∞(z, params)
 #   S_ic(x, y, z) = 2/pi*(S∞(x,params)-S_i)*atan((Lz-z)/delta_B*Le^(1/3))+S_i
    c_ic(x, y, z) = 0
    d_ic(x, y, z) = d∞(z, params)
    uᵢ = 0.005*rand(size(u)...)
    vᵢ = 0.005*rand(size(v)...)
    wᵢ = 0.005*rand(size(w)...)

    uᵢ .-= mean(uᵢ)
    vᵢ .-= mean(vᵢ)
    wᵢ .-= mean(wᵢ)
    uᵢ .+= params.u_b
    vᵢ .+= params.v_b    

 
#    plumewidth(x)=.0833*x;
#    umax=.04
   
#function u_ic(x, y, z)

#    if z > Lz-plumewidth(x)
#        return 5.77*x*umax*(1-(Lz-z)/plumewidth(x))^6*((Lz-z)/plumewidth(x))^(1/2)
#    else
#        return 0.0
#    end
#end
    
# uᵢ .+=  u_ic(x,y,z)

    set!(model, c=c_ic, d=d_ic, u=uᵢ, v=vᵢ, w=wᵢ, T=T_ic, S=S_ic)
end
#----

#++++ Outputs
@info "Creating output fields"

# y-component of vorticity
ω_y = Field(∂z(u) - ∂x(w))

outputs = (; u, v, w, T, S, c, d, ω_y)

if mass_flux
    saved_output_prefix = "iceplume"
else
    saved_output_prefix = "iceplume_nomf"
end
saved_output_filename = saved_output_prefix * ".nc"
checkpointer_prefix = "checkpoint_" * saved_output_prefix

#++++ Check for checkpoints
if any(startswith("$(checkpointer_prefix)_iteration"), readdir(rundir))
    @warn "Checkpoint $saved_output_prefix found. Assuming this is a pick-up simulation! Setting `overwrite_existing=false`."
    overwrite_existing = false
else
    @warn "No checkpoint for $saved_output_prefix found. Setting `overwrite_existing=true`."
    overwrite_existing = true
end
#----


simulation.output_writers[:fields] = NetCDFOutputWriter(model, outputs, 
                                                        schedule = TimeInterval(0.5days),
                                                        filename = saved_output_filename,
                                                        overwrite_existing = overwrite_existing)


ccc_scratch = Field{Center, Center, Center}(model.grid) # Create some scratch space to save memory

uv = Field((@at (Center, Center, Center) u*v))
uw = Field((@at (Center, Center, Center) u*w))
vw = Field((@at (Center, Center, Center) v*w))
uT = Field((@at (Center, Center, Center) u*T))
vT = Field((@at (Center, Center, Center) v*T))
wT = Field((@at (Center, Center, Center) w*T))
uS = Field((@at (Center, Center, Center) u*S))
vS = Field((@at (Center, Center, Center) v*S))
wS = Field((@at (Center, Center, Center) w*S))

#simulation.output_writers[:particle_writer] =
#NetCDFOutputWriter(model, model.particles, filename="particles.nc", schedule=TimeInterval(5seconds))


simulation.output_writers[:surface_slice_writer] =
    NetCDFOutputWriter(model, (; u,v,w, T,S, c, d), filename="face30.nc",
                       schedule=TimeInterval(4320), indices=(30, :, :),
                        overwrite_existing = overwrite_existing)

simulation.output_writers[:surface_slice_writer2] =
    NetCDFOutputWriter(model, (; u,v,w, T,S, c, d), filename="face1.nc",
                       schedule=TimeInterval(4320), indices=(1, :, :),
                        overwrite_existing = overwrite_existing)

simulation.output_writers[:surface_slice_writer3] =
    NetCDFOutputWriter(model, (; u,v,w, T,S, c, d), filename="face100.nc",
                       schedule=TimeInterval(4320), indices=(100, :, :),
                        overwrite_existing = overwrite_existing)

simulation.output_writers[:surface_slice_writer4] =
    NetCDFOutputWriter(model, (; u,v,w, T,S, c, d), filename="face200.nc",
                       schedule=TimeInterval(4320), indices=(200, :, :),
                        overwrite_existing = overwrite_existing)

simulation.output_writers[:surface_slice_writer5] =
    NetCDFOutputWriter(model, (; u,v,w, T,S,c, d), filename="top.nc",
                       schedule=TimeInterval(4320), indices=(:, :, Nz),
                        overwrite_existing = overwrite_existing)
#simulation.output_writers[:surface_slice_writer2] =
#    NetCDFOutputWriter(model, (; u,v,w, T,S), filename="neartop.nc",
#                       schedule=TimeInterval(10), indices=(:, :, round(params.Nz*3/4)),
#                        overwrite_existing = overwrite_existing)

simulation.output_writers[:y_slice_writer] =
    NetCDFOutputWriter(model,(; u, v, w, T,S,c,d ), filename="midy.nc",
                       schedule=TimeInterval(4320), indices=(:, 40, :), 
                       overwrite_existing = overwrite_existing)

simulation.output_writers[:surface_slice_writer6] =
    NetCDFOutputWriter(model, (; u,v,w, T,S,c, d), filename="z48.nc",
                       schedule=TimeInterval(4320), indices=(:, :, 48),
                        overwrite_existing = overwrite_existing)

simulation.output_writers[:surface_slice_writer7] =
    NetCDFOutputWriter(model, (; u,v,w, T,S,c, d), filename="z45.nc",
                       schedule=TimeInterval(4320), indices=(:, :, 45),
                        overwrite_existing = overwrite_existing)

simulation.output_writers[:surface_slice_writer8] =
    NetCDFOutputWriter(model, (; u,v,w, T,S,c, d), filename="z40.nc",
                       schedule=TimeInterval(4320), indices=(:, :, 40),
                        overwrite_existing = overwrite_existing)


output_interval=4320seconds
simulation.output_writers[:averages] = NetCDFOutputWriter(model, (; u, v, w, T, S,c,d, uv, uw, vw, uT, vT, wT, uS, vS, wS ),
                                                          schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                          filename = "timeavgedfields.nc",
                                                          overwrite_existing = overwrite_existing)

KE = KineticEnergy(model)
ε = KineticEnergyDissipationRate(model)
∫KE = Integral(KE)
∫ε = Integral(ε)
#∫εᴰ = Integral(KineticEnergyDiffusiveTerm(model))

#KE_output_fields = (; KE, ε, ∫KE, ∫ε, ∫εᴰ)
#simulation.output_writers[:nc] = NetCDFOutputWriter(model, KE_output_fields,
#                                                    filename = "KE_fields.nc",
#                                                    schedule = TimeInterval(20second),
#                                                    overwrite_existing = overwrite_existing)


simulation.output_writers[:checkpointer] = Checkpointer(model, 
                                                        schedule = TimeInterval(4320seconds), 
                                                        prefix = checkpointer_prefix,
                                                        cleanup = true,
                                                        )
#----

#++++ Ready to press the big red button:
run!(simulation; pickup=true)
#----

