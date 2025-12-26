"use client"

import { cn } from "@/lib/utils"
import { motion } from "framer-motion"

interface ShiningTextProps {
  text: string
  className?: string
  shimmerColor?: string
}

export const ShiningText: React.FC<ShiningTextProps> = ({
  text,
  className,
  shimmerColor = "#ffffff",
}) => {
  return (
    <motion.span
      className={cn(
        "relative inline-block overflow-hidden",
        className
      )}
      initial={{ opacity: 0.7 }}
      animate={{ opacity: [0.7, 1, 0.7] }}
      transition={{
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    >
      <span className="relative z-10">{text}</span>
      <motion.span
        className="absolute inset-0 z-20"
        style={{
          background: `linear-gradient(90deg, transparent 0%, ${shimmerColor}40 50%, transparent 100%)`,
        }}
        initial={{ x: "-100%" }}
        animate={{ x: "100%" }}
        transition={{
          duration: 1.5,
          repeat: Infinity,
          ease: "easeInOut",
          repeatDelay: 0.5,
        }}
      />
    </motion.span>
  )
}

export default ShiningText

