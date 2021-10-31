//
// Created by cjg on 10/31/21.
//

#include <clang-c/Index.h>

#include <iostream>

class Generator {
public:
    void Generate();

private:
    static CXChildVisitResult visit(CXCursor cursor, CXCursor parent, CXClientData client_data);
    CXChildVisitResult visit(CXCursor cursor, CXCursor parent);
};

void Generator::Generate() {
    char *args[] = { "-I/usr/local/cuda/include" };
    auto index = clang_createIndex(0, 0);
    auto tu = clang_createTranslationUnitFromSourceFile(index, "test.cpp", 1, args,
                                                        0, nullptr);
    clang_visitChildren(clang_getTranslationUnitCursor(tu),
                        Generator::visit, this);
    clang_disposeTranslationUnit(tu);
}

CXChildVisitResult Generator::visit(CXCursor cursor, CXCursor parent, CXClientData client_data) {
    return reinterpret_cast<Generator *>(client_data)->visit(cursor, parent);
}

CXChildVisitResult Generator::visit(CXCursor cursor, CXCursor parent) {
    auto kind = clang_getCursorKind(cursor);
    auto name = clang_getCursorDisplayName(cursor);
    if (kind == CXCursor_FunctionDecl) {
        if (clang_isCursorDefinition(cursor)) {
            // ignoring local function definitions
            return CXChildVisit_Continue;
        }
        name = clang_Cursor_getMangling(cursor);
        std::cout << "Function: " << reinterpret_cast<const char *>(name.data) << std::endl;
        auto resultType = clang_getCursorResultType(cursor);
        name = clang_getTypeSpelling(resultType);
        std::cout << "  Return: " << reinterpret_cast<const char *>(name.data) << std::endl;
    } else if (kind == CXCursor_ParmDecl) {
        auto indirections = 0;
        CXType type = clang_getCursorType(cursor);
        while (true) {
            type = clang_getCanonicalType(type);
            if (!(type.kind & CXType_Pointer)) {
                break;
            }
            type = clang_getPointeeType(type);
            indirections++;
        }
        auto typeSpelling = clang_getTypeSpelling(type);
        std::string typeName(reinterpret_cast<const char *>(typeSpelling.data));
        if (typeName.empty()) {
            typeName = "void";
        }
        std::cout << "  Param: " << typeName;
        for (auto i = 0; i < indirections; i++) {
            std::cout << "*";
        }
        std::cout << " " << reinterpret_cast<const char *>(name.data) << std::endl;
    } else if (kind == CXCursor_StructDecl && clang_isCursorDefinition(cursor)) {
        std::cout << "Struct: " << reinterpret_cast<const char *>(name.data) << std::endl;
    } else if (kind == CXCursor_FieldDecl) {
        auto type = clang_getCursorType(cursor);
        type = clang_getCanonicalType(type);
        auto typeName = clang_getTypeSpelling(type);
        std::cout << "  Field: " << reinterpret_cast<const char *>(typeName.data) << " "
        << reinterpret_cast<const char *>(name.data) << std::endl;
    } else if (kind == CXCursor_EnumDecl) {
        std::cout << "Enum: " << reinterpret_cast<const char *>(name.data) << std::endl;
    }
    return CXChildVisit_Recurse;
}

int main() {
    Generator g;
    g.Generate();
}
