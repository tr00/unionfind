
all:
	clang -O3 -flto -march=skylake bench.c uf32.c				-o ooo
	clang -O3 -flto -march=skylake bench.c uf32.c -DA			-o aoo
	clang -O3 -flto -march=skylake bench.c uf32.c     -DS		-o oso
	clang -O3 -flto -march=skylake bench.c uf32.c -DA -DS		-o aso
	clang -O3 -flto -march=skylake bench.c uf32.c         -DN	-o oon
	clang -O3 -flto -march=skylake bench.c uf32.c -DA     -DN	-o aon
	clang -O3 -flto -march=skylake bench.c uf32.c     -DS -DN	-o osn
	clang -O3 -flto -march=skylake bench.c uf32.c -DA -DS -DN	-o asn

run:
#	./ooo
	./aoo
#	./oso
	./aso
#	./oon
	./aon
#	./osn
	./asn

clean:
	rm -r ooo
	rm -r aoo
	rm -r oso
	rm -r aso
	rm -r oon
	rm -r aon
	rm -r osn
	rm -r asn