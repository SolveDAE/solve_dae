! Modified particle on a circular track subject to tangential force, see Arevalo1995.
!
! This driver is meant to be linked against E. Hairer's RADAU5 code
! (radau5.f, decsol.f, dc_decsol.f), copied verbatim into ../radau5/ so that
! this shared solver code can be reused by other *_radau5.f90 drivers
! alongside this one. Build with, e.g.:
!
!     gfortran -O2 -o particle_radau5 \
!         ../radau5/radau5.f ../radau5/decsol.f ../radau5/dc_decsol.f \
!         particle_radau5.f90
!     ./particle_radau5
!
! or via CMake: from the benchmarks/ directory,
!     cmake -S . -B build && cmake --build build --target particle_radau5
!     ./build/particle_radau5
!
! -----------------------------------------------------------------------
! Why a *different* formulation than particle.c / particle.py
! -----------------------------------------------------------------------
! particle.c/particle.py solve the DAE in residual form
!
!     0 = R(t, y, y')
!
! using the "stabilized index 1" reformulation of Anantharaman1991 (as used
! by Hiller/IDA): two auxiliary states La(t), Mu(t) are introduced whose
! time *derivatives* La'=lambda, Mu'=mu act as the actual Lagrange
! multipliers. This raises the effective differentiation index back down
! to 1, which is exactly what IDA (a residual-based, index-1 solver) needs.
!
! RADAU5 instead expects a *linearly implicit* system
!
!     M y' = f(t, y)
!
! with (possibly singular, but *constant*) mass matrix M. The auxiliary
! "integrate the multiplier" trick above is not of this form (the
! multipliers only ever appear differentiated), so it cannot be handed to
! RADAU5 directly. What *can* be handed to RADAU5 is the classical GGL
! (Gear-Gupta-Leimkuhler) stabilized-index-2 formulation, in which the
! multipliers lambda, mu appear undifferentiated as genuine algebraic
! unknowns:
!
!     x'  = u + 2*x*mu
!     y'  = v + 2*y*mu
!     u'  = 2*x*lambda - y*force(t)
!     v'  = 2*y*lambda + x*force(t)
!     0   = 2*(x*u + y*v)        ! velocity constraint (stabilizes index 3 -> 2)
!     0   = x**2 + y**2 - 1      ! position constraint
!
! With y = (x, y, u, v, lambda, mu) this is exactly M y' = f(y) with
!
!     M = diag(1, 1, 1, 1, 0, 0)   (singular, constant -> IMAS=1)
!
! and f as coded in FCN below. Differentiating the two algebraic rows once
! shows that BOTH lambda and mu are determined after a single
! differentiation (using x**2+y**2=1 the position-constraint row also
! collapses to an explicit equation for mu) -- i.e. both are index-2
! variables, not index-3. This is the "stabilized index two" formulation
! the task refers to, and it is what is implemented below.
!
! -----------------------------------------------------------------------
! Do the multiplier errors need to be scaled by h?
! -----------------------------------------------------------------------
! Yes for RADAU5's *internal* error control, and this is mandatory, not
! optional. RADAU5 keeps a single weight vector SCAL(1:N) that is used both
! for the Newton-iteration convergence test (DYNO in RADCOR) and for the
! local error estimate (ESTRAD). If IWORK(5),IWORK(6),IWORK(7) mark
! variables NIND1+1..NIND1+NIND2 as index-2, RADAU5 divides SCAL for those
! components by HHFAC (~h) before both of these norms are formed (see
! radau5.f, routine RADCOR, right after "IF (INDEX2) ... SCAL(I)=SCAL(I)/HHFAC").
! This reflects the standard DAE-order-reduction fact (Hairer & Wanner,
! "Solving ODEs II", Sec. VII.4-5): the achievable local accuracy of an
! index-2 algebraic variable is inherently one power of h weaker than that
! of the index-1/differential variables, so testing it against the same
! *unscaled* tolerance would either be hopelessly pessimistic (wasting
! steps) or silently under-resolved. Concretely for this problem:
!
!     IWORK(5) = NIND1 = 4   ( x, y, u, v )
!     IWORK(6) = NIND2 = 2   ( lambda, mu )
!     IWORK(7) = NIND3 = 0
!
! and the ordering of Y must match (index-1 vars first, then index-2), as
! is already the case above. Leaving IWORK(5..7) at their default (0 ->
! NIND1=N, i.e. "treat everything as index 1") is wrong for this problem
! and defeats RADAU5's step-size/Newton logic for lambda, mu.
!
! For *reporting* the error against the known reference solution (as done
! below), no extra h-scaling should be added by hand: we just compare
! lambda, mu componentwise against their true values, exactly as for x, y,
! u, v. What the h-scaling above predicts is the *slope* you should expect
! to see in those curves: because lambda and mu are index-2 quantities,
! their error typically converges with one order less in h than x,y,u,v --
! that shows up naturally in the reported numbers and is expected, correct
! DAE behaviour, not a bug in the driver.
! -----------------------------------------------------------------------

program particle_radau5
    implicit double precision (a-h,o-z)
    integer, parameter :: nd = 6
    ! LWORK for full Jacobian (LJAC=N) and full, non-identity mass matrix
    ! (LMAS=N, LE=N): LWORK >= N*(LJAC+LMAS+3*LE+12)+20 = 5*N*N+12*N+20.
    integer, parameter :: lwork  = 5*nd*nd + 12*nd + 20
    integer, parameter :: liwork = 3*nd + 20

    double precision :: y(nd), ytrue(nd), work(lwork)
    integer          :: iwork(liwork)
    double precision :: rtol(1), atol(1), rpar(1)
    integer          :: ipar(1)

    double precision :: omega
    common /partprm/ omega

    external :: fcn, jacpart, maspart, soloutdummy

    integer          :: m, mmax, i
    double precision :: pi, t0, t1, x, xend, h, rt
    double precision :: t_start, t_end, elapsed
    double precision :: err_state, err_la, err_mu

    pi    = 3.14159265358979324d0
    omega = 2.0d0*pi

    t0   = 0.0d0
    t1   = 2.0d0*pi
    mmax = 40

    open(unit=10, file='particle_errors_RADAU5.csv', status='replace')
    write(10,'(A)') 'rtol,elapsed_time,error_state,error_la,error_mu,'// &
                     'nsteps,naccpt,nrejct,nfcn,njac'

    do m = 0, mmax
        rt = 10.0d0**( -(3.0d0 + dble(m)/4.0d0) )

        itol    = 0
        rtol(1) = rt
        atol(1) = rt

        x    = t0
        xend = t1
        h    = 1.0d-2*rt

        call truesol(x, y)

        ijac  = 1
        mljac = nd
        mujac = nd
        imas  = 1
        mlmas = nd
        mumas = nd
        iout  = 0

        work  = 0.0d0
        iwork = 0
        ! index-1 / index-2 / index-3 split, see header comment above
        iwork(5) = 4
        iwork(6) = 2
        iwork(7) = 0

        call cpu_time(t_start)
        call radau5(nd, fcn, x, y, xend, h,          &
                     rtol, atol, itol,                 &
                     jacpart, ijac, mljac, mujac,       &
                     maspart, imas, mlmas, mumas,       &
                     soloutdummy, iout,                 &
                     work, lwork, iwork, liwork, rpar, ipar, idid)
        call cpu_time(t_end)
        elapsed = t_end - t_start

        if (idid /= 1) then
            write(*,*) 'RADAU5 failed, idid=', idid, ' at m=', m
        end if

        call truesol(t1, ytrue)

        err_state = 0.0d0
        do i = 1, 4
            err_state = err_state + (y(i)-ytrue(i))**2
        end do
        err_state = sqrt(err_state)
        err_la = abs(y(5)-ytrue(5))
        err_mu = abs(y(6)-ytrue(6))

        write(*,900) rt, elapsed, err_state, err_la, err_mu
900     format('rtol=',es10.3,'  time=',es10.3,'  err_state=',es10.3, &
               '  err_la=',es10.3,'  err_mu=',es10.3)

        write(10,910) rt, elapsed, err_state, err_la, err_mu, &
                       iwork(16), iwork(17), iwork(18), iwork(14), iwork(15)
910     format(es17.10,',',es17.10,',',es17.10,',',es17.10,',',es17.10, &
               ',',i8,',',i8,',',i8,',',i8,',',i8)

    end do

    close(10)

end program particle_radau5


subroutine truesol(t, y)
    ! Reference solution of the circular-track particle problem, expressed
    ! in the GGL / stabilized-index-2 variables y = (x, y, u, v, lambda, mu).
    ! lambda(t) = -phi_p(t)**2/2 follows from the u', v' equations; mu(t)=0
    ! since the exact trajectory never needs a position-drift correction.
    implicit double precision (a-h,o-z)
    double precision :: y(6)
    double precision :: omega
    common /partprm/ omega
    double precision :: phi, phip

    phi  = omega*sin(t)
    phip = omega*cos(t)

    y(1) = cos(phi)
    y(2) = sin(phi)
    y(3) = -sin(phi)*phip
    y(4) =  cos(phi)*phip
    y(5) = -0.5d0*phip*phip
    y(6) = 0.0d0

    return
end subroutine truesol


subroutine fcn(n, x, y, f, rpar, ipar)
    ! RHS f(t,y) of  M y' = f(t,y),  y = (x, y, u, v, lambda, mu).
    implicit double precision (a-h,o-z)
    dimension y(n), f(n), rpar(*)
    integer ipar(*)
    double precision :: omega
    common /partprm/ omega
    double precision :: posx, posy, velu, velv, lam, mu, force

    posx = y(1)
    posy = y(2)
    velu = y(3)
    velv = y(4)
    lam  = y(5)
    mu   = y(6)

    force = -omega*sin(x)   ! = phi''(t)

    f(1) = velu + 2.0d0*posx*mu
    f(2) = velv + 2.0d0*posy*mu
    f(3) = 2.0d0*posx*lam - posy*force
    f(4) = 2.0d0*posy*lam + posx*force
    f(5) = 2.0d0*(posx*velu + posy*velv)
    f(6) = posx*posx + posy*posy - 1.0d0

    return
end subroutine fcn


subroutine jacpart(n, x, y, dfy, ldfy, rpar, ipar)
    ! Analytic Jacobian DFY(i,j) = d f_i / d y_j of the FCN above.
    implicit double precision (a-h,o-z)
    dimension y(n), dfy(ldfy,n), rpar(*)
    integer ipar(*)
    double precision :: omega
    common /partprm/ omega
    double precision :: posx, posy, velu, velv, lam, mu, force
    integer :: i, j

    posx = y(1)
    posy = y(2)
    velu = y(3)
    velv = y(4)
    lam  = y(5)
    mu   = y(6)

    force = -omega*sin(x)

    do j = 1, n
        do i = 1, n
            dfy(i,j) = 0.0d0
        end do
    end do

    dfy(1,1) = 2.0d0*mu
    dfy(1,3) = 1.0d0
    dfy(1,6) = 2.0d0*posx

    dfy(2,2) = 2.0d0*mu
    dfy(2,4) = 1.0d0
    dfy(2,6) = 2.0d0*posy

    dfy(3,1) = 2.0d0*lam
    dfy(3,2) = -force
    dfy(3,5) = 2.0d0*posx

    dfy(4,1) = force
    dfy(4,2) = 2.0d0*lam
    dfy(4,5) = 2.0d0*posy

    dfy(5,1) = 2.0d0*velu
    dfy(5,2) = 2.0d0*velv
    dfy(5,3) = 2.0d0*posx
    dfy(5,4) = 2.0d0*posy

    dfy(6,1) = 2.0d0*posx
    dfy(6,2) = 2.0d0*posy

    return
end subroutine jacpart


subroutine maspart(n, am, lmas, rpar, ipar)
    ! Constant, singular mass matrix M = diag(1,1,1,1,0,0).
    implicit double precision (a-h,o-z)
    dimension am(lmas,n), rpar(*)
    integer ipar(*)
    integer :: i, j

    do j = 1, n
        do i = 1, n
            am(i,j) = 0.0d0
        end do
    end do

    am(1,1) = 1.0d0
    am(2,2) = 1.0d0
    am(3,3) = 1.0d0
    am(4,4) = 1.0d0

    return
end subroutine maspart


subroutine soloutdummy(nr, xold, x, y, cont, lrc, n, rpar, ipar, irtrn)
    ! Never called since IOUT=0 in the RADAU5 call above.
    implicit double precision (a-h,o-z)
    dimension y(n), cont(lrc), rpar(*)
    integer ipar(*)
    irtrn = 0
    return
end subroutine soloutdummy
