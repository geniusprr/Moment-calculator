declare module "react-katex" {
  import type { ComponentType } from "react";

  export interface KatexBaseProps {
    math: string;
    errorColor?: string;
    renderError?: (error: Error) => React.ReactNode;
  }

  export const BlockMath: ComponentType<KatexBaseProps>;
  export const InlineMath: ComponentType<KatexBaseProps>;
}