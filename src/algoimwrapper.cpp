#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"
#include "jlcxx/tuple.hpp"
#include "jlcxx/const_array.hpp"

#include "algoim/hocp.hpp"
#include "algoim/utility.hpp"
#include "algoim/quadrature_general.hpp"
#include "algoim/quadrature_multipoly.hpp"

using namespace algoim;
using algoim::util::sqr;

struct LevelSetFunction {};

template<int N>
struct SafeCFunctionLevelSet : LevelSetFunction
{

    jlcxx::SafeCFunction op_fun;
    jlcxx::SafeCFunction grad_fun;

    SafeCFunctionLevelSet(
      jlcxx::SafeCFunction f,
      jlcxx::SafeCFunction g
    ) : op_fun(f),
        grad_fun(g)
    {
    }

    real value(const uvector<real,N>& x, float id) const
    {
        auto f = jlcxx::make_function_pointer<real(const uvector<real,N> &, float)>(op_fun);
        return f(x,id);
    }

    uvector<real,N> gradient(const uvector<real,N>& x, float id) const
    {
        auto f = jlcxx::make_function_pointer<const uvector<real,N> &(const uvector<real,N> &, float)>(grad_fun);
        return f(x,id);
    }

};

template<int N, typename T, typename F>
void fill_quad_data( const F& fphi,                                      \
                     jlcxx::ArrayRef<T> jx,    jlcxx::ArrayRef<T> jw,    \
                     jlcxx::ArrayRef<T> jxmin, jlcxx::ArrayRef<T> jxmax, \
                     int q, int phase, float id )
{
    
    uvector<int,N> P = q;
    uvector<T,N> xmin, xmax;
    for (int i = 0; i < N; ++i)
    {
        xmin(i) = jxmin[i];
        xmax(i) = jxmax[i];
    }
    
    // Construct phi by mapping [0,1] onto bounding box [xmin,xmax]
    xarray<T,N> phi(nullptr, P);
    algoim_spark_alloc(T, phi);

    auto value = [&](const uvector<real,N>& x, float id) { return fphi.value(x,id); };
    bernstein::bernsteinInterpolate<N>([&](const uvector<T,N>& x) { return value(xmin + x * (xmax - xmin),id); }, phi);

    // Build quadrature hierarchy
    ImplicitPolyQuadrature<N> ipquad(phi);

    // Compute quadrature scheme and record the nodes & weights; phase0 corresponds to
    // {phi < 0}, phase1 corresponds to {phi > 0}, and surf corresponds to {phi == 0}.
    std::vector<uvector<T,N+1>> quad;
    if ( phase == 0 ){
      ipquad.integrate_surf(AutoMixed, q, [&](const uvector<T,N>& x, T w, const uvector<T,N>& wn)
      {
          quad.push_back(add_component(x, N, w));
      });
    }
    else {
      ipquad.integrate(AutoMixed, q, [&](const uvector<T,N>& x, T w)
      {
          if ( bernstein::evalBernsteinPoly(phi, x) * phase > 0)
              quad.push_back(add_component(x, N, w));
      });
    }

    // Fill coords and weights of quadrature in Cpp-to-Julia array data
    int np = size(quad);
    // std::cout << np << std::endl;
    for (int i = 0; i < np; ++i) {
      const T* d = quad[i].data();
      // std::cout << d << std::endl;
      for (int j = 0; j < N; ++j)
        jx.push_back(d[j]);
      jw.push_back(d[N]);
    }

}

// LexicographicLoop is essentially an N-dimensional iterator for looping over the
// coordinates of a Cartesian grid having indices min(0) <= i < max(0),
// min(1) <= j < max(1), min(2) <= k < max(2), etc. The ordering is such that
// dimension 0 is inner-most, i.e., iterates the fastest, while dimension
// N-1 is outer-most and iterates the slowest.
template<int N>
class LexicographicLoop
{
    uvector<int,N> i;
    const uvector<int,N> min, max;
    bool valid;
public:
    LexicographicLoop(const uvector<int,N>& min, const uvector<int,N>& max)
        : i(min), min(min), max(max), valid(all(min < max))
    {}

    LexicographicLoop& operator++()
    {
        for (int dim = 0; dim <= N - 1; ++dim)
        {
            if (++i(dim) < max(dim))
                return *this;
            i(dim) = min(dim);
        }
        valid = false;
        return *this;
    }

    const uvector<int,N>& operator()() const
    {
        return i;
    }

    int operator()(int index) const
    {
        return i(index);
    }

    bool operator~() const
    {
        return valid;
    }
};

// A simple test functor whose purpose is to simulate a grid-defined scalar array
template<int N, typename T, typename Test>
struct TestFunctor
{
    const Test& test;
    const uvector<int,N> n;
    const uvector<T,N> dx;
    const uvector<T,N> xmin;

    TestFunctor(const Test& test, const uvector<int,N>& n, const uvector<T,N>& dx, const uvector<T,N>& xmin) : test(test), n(n), dx(dx), xmin(xmin) {}
    T operator() (const uvector<int,N>& i) const
    {
        uvector<int,N> j = i;
        float id = 1;
        for (int dim = 0; dim < N; ++dim)
        {
            if (j(dim) < 0)
                j(dim) = 0;
            else if (j(dim) >= n(dim))
                j(dim) = n(dim) - 1;
            id += j(dim)*(pow(n(dim),dim));
        }
        return test.value(j*dx + xmin, id);
    }
};

template<int N, int Degree, typename T, typename F>
void fill_cpp_data( const F& fphi, jlcxx::ArrayRef<int> partition,      \
                    jlcxx::ArrayRef<T> jxmin, jlcxx::ArrayRef<T> jxmax, \
                    jlcxx::ArrayRef<T> jxcpp )
{

    // Determine the type of polynomial to use based on given Degree and dimension N
    typedef typename algoim::StencilPoly<N,Degree>::T_Poly Poly;

    // Fill grid data
    uvector<int,N> n, ext;
    uvector<T,N> xmin, dx;
    for (int i = 0; i < N; ++i)
    {
        n(i)    =   partition[i];
        ext(i)  =   n(i) + 1;
        xmin(i) =   jxmin[i];
        dx(i)   = ( jxmax[i] - xmin(i) ) / n(i);
    }

    // Create a functor whose purpose is to simulate a n-dimensional scalar array
    TestFunctor<N,T,F> functor(fphi, n, dx, xmin);

    // Find all cells containing the interface and construct the high-order polynomials
    std::vector<algoim::detail::CellPoly<N,Poly>> cells;
    algoim::detail::createCellPolynomials(ext, functor, dx, true, cells);

    // Using the polynomials, sample the zero level set in each cell to create a cloud of seed points
    std::vector<uvector<T,N>> points;
    std::vector<int> pointcells;
    int subcellExt = 2;
    algoim::detail::samplePolynomials(cells, subcellExt, dx, xmin, points, pointcells);

    // Construct a k-d tree from the seed points
    algoim::KDTree<T,N> kdtree(points);

    // Pass everything to the closest point computation engine
    algoim::ComputeHighOrderCP<N,Poly> hocp(std::numeric_limits<double>::max(), // bandradius = infinity
        0.5*max(dx), // amount of overlap, i.e. size of bounding ball in Newton's method
        sqr(std::max(1.0e-14, pow(max(dx), Poly::order))), // tolerance to determine convergence
        cells, kdtree, points, pointcells, dx, xmin);

    // Loop over every grid point of domain
    for (LexicographicLoop<N> i(0, ext); ~i; ++i)
    {
        uvector<double,N> x = i()*dx + xmin;
        uvector<double,N> cp;

        // Compute the closest point to x
        hocp.compute(x, cp);

        for (int j = 0; j < N; ++j)
          jxcpp.push_back(cp(j));

    }

}

template<typename T, int N>
uvector<T,N> to_uvector( jlcxx::ArrayRef<T> jx )
{
    uvector<T,N> x;
    for (int i = 0; i < N; ++i)
        x(i) = jx[i];
    return x;
}

namespace jlcxx
{

    template<typename T, int N> struct IsMirroredType<uvector<T,N>> : std::false_type { };

    template<typename T, int N>
    struct BuildParameterList<uvector<T,N>>
    {
      typedef ParameterList<T,std::integral_constant<int64_t, N>> type;
    };

    template<> struct IsMirroredType<LevelSetFunction> : std::false_type { };

    template<int N> struct SuperType<SafeCFunctionLevelSet<N>> { typedef LevelSetFunction type; };
    template<int N> struct IsMirroredType<SafeCFunctionLevelSet<N>> : std::false_type { };

}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{

    using namespace jlcxx;
    using namespace algoim;

    mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>,jlcxx::TypeVar<2>>>("AlgoimUvector")
      .apply<uvector<int,2>,uvector<int,3>,uvector<real,2>,uvector<real,3>>([](auto wrapped)
      {
        typedef typename decltype(wrapped)::type WrappedT;
        wrapped.module().set_override_module(jl_base_module);
        wrapped.method("length", [](const WrappedT& u) -> int64_t
        {
          return algoim::detail::extent_of<WrappedT>::value;
        });
        wrapped.method("*", [](real a, const WrappedT& u) {
          WrappedT v;
          int n = algoim::detail::extent_of<WrappedT>::value;
          for (int i = 0; i < n; ++i)
              v(i) = a*u(i);
          return v; 
        });
        wrapped.method("*", [](const WrappedT& u, real a) {
          WrappedT v;
          int n = algoim::detail::extent_of<WrappedT>::value;
          for (int i = 0; i < n; ++i)
              v(i) = u(i)*a;
          return v; 
        });
        wrapped.module().unset_override_module();
        wrapped.method("sqrnorm", [](const WrappedT& u) { return sqrnorm(u); });
        wrapped.method("data_array", [](const WrappedT& u) { return u.data(); });
      });

    mod.method("to_2D_uvector", &to_uvector<real,2>);
    mod.method("to_3D_uvector", &to_uvector<real,3>);

    mod.add_type<LevelSetFunction>("LevelSetFunction");
    
    // Map every C++ level set struct into a Julia type

    mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("SafeCFunctionLevelSet",jlcxx::julia_base_type<LevelSetFunction>())
      .apply<SafeCFunctionLevelSet<2>,SafeCFunctionLevelSet<3>>([](auto wrapped)
      {
        wrapped.template constructor<jlcxx::SafeCFunction,jlcxx::SafeCFunction>();
      });

    mod.method("fill_quad_data_cpp", &fill_quad_data<2,real,SafeCFunctionLevelSet<2>>);
    mod.method("fill_quad_data_cpp", &fill_quad_data<3,real,SafeCFunctionLevelSet<3>>);

    mod.method("fill_cpp_data_taylor_2", &fill_cpp_data<2,2,real,SafeCFunctionLevelSet<2>>);
    mod.method("fill_cpp_data_taylor_2", &fill_cpp_data<3,2,real,SafeCFunctionLevelSet<3>>);

    mod.method("fill_cpp_data_taylor_3", &fill_cpp_data<2,3,real,SafeCFunctionLevelSet<2>>);
    mod.method("fill_cpp_data_taylor_3", &fill_cpp_data<3,3,real,SafeCFunctionLevelSet<3>>);

    mod.method("fill_cpp_data_taylor_4", &fill_cpp_data<2,4,real,SafeCFunctionLevelSet<2>>);
    mod.method("fill_cpp_data_taylor_4", &fill_cpp_data<3,4,real,SafeCFunctionLevelSet<3>>);

    mod.method("fill_cpp_data_taylor_5", &fill_cpp_data<2,5,real,SafeCFunctionLevelSet<2>>);
    mod.method("fill_cpp_data_taylor_5", &fill_cpp_data<3,5,real,SafeCFunctionLevelSet<3>>);

    mod.method("fill_cpp_data_cubic", &fill_cpp_data<2,-1,real,SafeCFunctionLevelSet<2>>);
    mod.method("fill_cpp_data_cubic", &fill_cpp_data<3,-1,real,SafeCFunctionLevelSet<3>>);

}