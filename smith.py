# -*- coding: utf-8 -*-
import numpy as np
import sys

from typing import List, Tuple

class PolyReader:
    """
    Loads the data for a polynomial matrix from a text file
    """

    def __init__(self, inputFile: str):
        """
        Constructor which takes in the name of the file containing the input
        matrix

        :params inputFile: input file name
        """
        self.inputFile = inputFile

    def __processEntry(self, entry: str) -> np.poly1d:
        """
        Converts a string entry in the matrix into a numpy polynomial

        :param entry: string representation of a polynomial

        :returns: polynomial represented as a numpy poly1d
        """

        # Cleans the input
        entry = entry.replace(" ", "")
        entry = entry.replace("-", "+-")
        entry = entry.split("+")

        coefs = []
        for monomial in entry:
            if len(monomial) > 0:
                monomial = monomial.split("x")
                coef = 1
                if len(monomial[0]) != 0:
                    coef = float(monomial[0])

                exponent = 0
                if len(monomial) > 1:
                    if monomial[1]:
                        exponent = int(monomial[1])
                    else:
                        exponent = 1

                coefs.append((coef, exponent))

        coefs.sort(key = lambda x: x[1])
        poly = [0] * (coefs[-1][1] + 1)

        for coef in coefs:
            poly[coef[1]] += coef[0]

        return np.poly1d(poly[::-1])

    def __processLine(self, line: str) -> Tuple[List[np.poly1d], int]:
        """
        Converts a line from the matrix into an array of polynomial entries

        :param line: string representation of a line in the matrix

        :returns: tuple consisting of the list of polynomials and number of
                  columns read
        """
        line = line.replace("\n", "")
        line = line.split(",")
        nCol = len(line)
        row = []

        for entry in line:
            row.append(self.__processEntry(entry))

        return row, nCol

    def readMatrix(self) -> np.matrix:
        """
        Reads the matrix data from the file supplied

        :returns: numpy matrix containing the polynomials as represented by
                  numpy
        """
        pols = []
        nCol = None
        with open(self.inputFile, 'r') as fp:
            line = fp.readline().replace("\n","")
            while line:
                row, colsFound = self.__processLine(line)
                pols.append(row)

                if not nCol:
                    nCol = colsFound
                elif nCol != colsFound:
                    # Inconsistent number of columns
                    raise ValueError(f"Input contains rows with different "\
                                      "number of columns")

                line = fp.readline()

        if len(pols) != nCol:
            # Only square matrices are allowed
            raise ValueError("The number of rows and columns must be equal")

        result = np.empty((nCol, nCol), dtype=object)
        for i in range(len(pols)):
            for j in range(nCol):
                result[i, j] = pols[i][j]

        return result


class PolyWriter:
    """
    Writes the data for a polynomial matrix to the standard output
    """

    def printEntry(self, poly: np.poly1d):
        """
        Writes a polynomial represented as an np.poly1d to the standard output

        :param poly: polynomial entry to be written out
        """
        strpoly = ""
        counter = poly.order
        coefs = poly.coeffs

        first = True

        for i in range(poly.order + 1):
            # print each monomial term, with highest order terms coming first
            if coefs[i] != 0:
                if not first:
                    strpoly += " "
                    if coefs[i] > 0:
                        strpoly += "+"

                if not (abs(coefs[i]) == 1 and i < poly.order):
                    strpoly += str(coefs[i])
                elif coefs[i] == -1:
                    strpoly += "-"

                if i < poly.order:
                    strpoly += "x"
                    if i < poly.order - 1:
                        strpoly += str(poly.order - i)

                first = False

        if strpoly == "":
            strpoly = "0"

        print(strpoly, end='')

    def printMatrix(self, matrix: np.matrix):
        """
        Writes a matrix of polynomials repesented as an np.matrix to the
        standard output

        :param matrix: matrix of polynomials to be written out
        """
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.printEntry(matrix[i, j])
                if j < matrix.shape[1] - 1:
                    print(", ", end='')
            print("")


class MatrixSolver:
    """
    Converts a matrix of polynomials into its Smith Normal Form
    """

    def __init__(self, matrix: np.matrix):
        """
        Constructor which takes in the input matrix

        :params matrix: matrix of polynomials to be factored
        """
        self.matrix = matrix

    def solve(self) -> Tuple[np.matrix, np.matrix, np.matrix]:
        """
        Calculates the Smith Normal Form for a matrix, and returns it with the
        row-transform and column-transform matrices associated with it

        :returns: P, S and Q, which are the row-transform, Smith Normal Form
                  and column-transform matrices, respectively
        """
        nRow, nCol = self.matrix.shape
        n = min(nRow, nCol)
        counter = 0

        writer = PolyWriter()

        P = np.empty((nRow, nRow), dtype=object)
        Q = np.empty((nCol, nCol), dtype=object)
        S = self.matrix

        for i in range(nRow):
            for j in range(nRow):
                P[i,j] = np.poly1d(0) if i != j else np.poly1d(1)
        for i in range(nCol):
            for j in range(nCol):
                Q[i,j] = np.poly1d(0) if i != j else np.poly1d(1)

        while counter < n:
            clearedPositions = False
            while not clearedPositions:
                position = (counter, counter)
                for i in range(counter, nRow):
                    for j in range(counter, nCol):
                        entry = S[i, j]
                        if not (entry.order == 0 and \
                                np.polyval(entry, 0) == 0):
                            if entry.order < S[position].order:
                                position = (i, j)

                clearedPositions = True
                for i in range(nRow):
                    checkPol = matrix[i, counter]
                    if (not (checkPol.order == 0 and\
                             np.polyval(checkPol, 0) == 0)) and i != counter:
                        clearedPositions = False
                for i in range(nRow):
                    checkPol = matrix[counter, i]
                    if (not (checkPol.order == 0 and \
                             np.polyval(checkPol, 0) == 0)) and i != counter:
                        clearedPositions = False

                if position == (counter, counter) and clearedPositions:
                    entry = S[counter, counter]
                    if entry.order == 1 and np.polyval(entry, 0) == 0:
                        counter = n
                    else:
                        if S[position].coef[0] != 0 and \
                           S[position].coef[0] != 1:
                            print("Divides row %d by %f" %
                                  (counter, S[position].coef[0]))
                            P[counter, :] /= S[position].coef[0]
                            S[position] /= S[position].coef[0]
                else:
                    if counter != position[0]:
                        print("Switch rows %d and %d" %
                              (counter + 1, position[0] + 1))
                    if counter != position[1]:
                        print("Switch columns %d and %d" %
                              (counter + 1, position[1] + 1))

                    # Row switching
                    aux = S[counter,:].copy()
                    S[counter,:] = S[position[0],:].copy()
                    S[position[0],:] = aux

                    # P matrix
                    aux = P[counter,:].copy()
                    P[counter,:] = P[position[0],:].copy()
                    P[position[0],:] = aux

                    # Column switching
                    aux = S[:,counter].copy()
                    S[:,counter] = S[:,position[1]].copy()
                    S[:,position[1]] = aux

                    # Q matrix
                    aux = Q[:,counter].copy()
                    Q[:,counter] = Q[:,position[1]].copy()
                    Q[:,position[1]] = aux

                    for i in range(counter + 1, nRow):
                        q, _ = np.polydiv(S[i, counter], S[counter, counter])
                        for j in range(nCol):
                            S[i,j] -= q * S[counter,j]

                            # P matrix
                            P[i,j] -= q * P[counter, j]

                        if q != np.poly1d(0):
                            print(f"Subtracts, from row {i+1}, column " +
                                  f"{counter+1} multiplied by (", end='')
                            writer.printEntry(q)
                            print(")")
                    for i in range(counter + 1, nCol):
                        q, _ = np.polydiv(S[counter, i], S[counter, counter])
                        for j in range(nRow):
                            S[j,i] -= q * S[j,counter]

                            # Q matrix
                            Q[j,i] -= q * Q[j,counter]

                        if q != np.poly1d(0):
                            print(f"Subtracts, from columns {i+1}, row " +
                                  f"{counter+1} mutiplied by (", end='')
                            writer.printEntry(q)
                            print(")")

            counter += 1

        return P, S, Q


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: smith.py <filename>")
    else:
        inputFile = sys.argv[1]
        reader = PolyReader(inputFile)
        matrix = reader.readMatrix()

        solver = MatrixSolver(matrix)
        print("Operations:")
        print()
        P, S, Q = solver.solve()

        writer = PolyWriter()
        print()
        print("PAQ = S")
        print()
        print("Row-transform matrix (P)")
        writer.printMatrix(P)
        print()
        print("Column-transform matrix (Q)")
        writer.printMatrix(Q)
        print()
        print("Matrix in Smith Normal Form (S)")
        writer.printMatrix(S)
