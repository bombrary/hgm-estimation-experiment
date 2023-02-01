loadPackage "Dmodules";

D3 = QQ[x, mu, sig,
       dx, dmu, dsig,
       WeylAlgebra=>{x=>dx, mu=>dmu, sig=>dsig}];
I = ideal(sig*dx + (x - mu),
          sig*dmu - (x - mu),
          2*sig^2*dsig + sig - (x - mu)^2);
J = WeylClosure I;
print(toString(J));
