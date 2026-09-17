import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

// Library build for external consumers (goldilocks-agent/app installs the
// packed output of this config as a normal npm dependency -- see src/index.ts
// and junwen94/goldilocks-agent#1). Deliberately separate from vite.config.ts,
// which stays exactly as-is for this repo's own app build/dev server/tests.
export default defineConfig({
  plugins: [
    react(),
    dts({
      tsconfigPath: "./tsconfig.app.json",
      include: ["src"],
      entryRoot: "src",
      rollupTypes: true,
    }),
  ],
  build: {
    // Separate from vite.config.ts's "dist" (the app build) -- both configs
    // would otherwise silently clobber each other's output since Vite
    // defaults every build to "dist".
    outDir: "dist-lib",
    cssCodeSplit: false,
    lib: {
      entry: fileURLToPath(new URL("./src/index.ts", import.meta.url)),
      formats: ["es"],
      fileName: () => "index.js",
    },
    rollupOptions: {
      external: ["react", "react-dom", "react/jsx-runtime"],
    },
  },
});
